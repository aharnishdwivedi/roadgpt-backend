package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"

	"github.com/labstack/echo/v4"
)

type TenderIQHandler struct {
	geminiService *GeminiService
	vectorStore   *VectorStore
	pdfParser     *PDFParser
}

type UploadResponse struct {
	DocumentID string                 `json:"document_id"`
	Filename   string                 `json:"filename"`
	Pages      int                    `json:"pages"`
	Metadata   map[string]interface{} `json:"metadata"`
	Message    string                 `json:"message"`
}

type AnalysisRequest struct {
	DocumentID string `json:"document_id"`
	Query      string `json:"query"`
}

type AnalysisResponse struct {
	DocumentID     string          `json:"document_id"`
	Query          string          `json:"query"`
	Analysis       TenderAnalysis  `json:"analysis"`
	RelevantChunks []SearchResult  `json:"relevant_chunks"`
	Message        string          `json:"message"`
}

type TenderAnalysis struct {
	TenderID            string                 `json:"tender_id"`
	Title               string                 `json:"title"`
	DueDate             string                 `json:"due_date"`
	IssuingAuthority    string                 `json:"issuing_authority"`
	ContractValue       string                 `json:"contract_value"`
	ProjectOverview     string                 `json:"project_overview"`
	FinancialReqs       FinancialRequirements  `json:"financial_requirements"`
	EligibilityHighlights []string             `json:"eligibility_highlights"`
	ImportantDates      ImportantDates         `json:"important_dates"`
	RiskAnalysis        RiskAnalysis           `json:"risk_analysis"`
}

type FinancialRequirements struct {
	ContractValue  string `json:"contract_value"`
	EMD           string `json:"emd"`
	PerformanceBG string `json:"performance_bg"`
	DocumentFees  string `json:"document_fees"`
}

type ImportantDates struct {
	PreBidQueries       string `json:"pre_bid_queries"`
	BidSubmission       string `json:"bid_submission"`
	TechnicalBidOpening string `json:"technical_bid_opening"`
	FinancialBidOpening string `json:"financial_bid_opening"`
}

type RiskAnalysis struct {
	PenaltyRisk string   `json:"penalty_risk"`
	Retention   string   `json:"retention"`
	KeyRisks    []string `json:"key_risks"`
}

type DocumentListResponse struct {
	Documents []DocumentInfo `json:"documents"`
}

type DocumentInfo struct {
	ID       string                 `json:"id"`
	Filename string                 `json:"filename"`
	Pages    int                    `json:"pages"`
	Metadata map[string]interface{} `json:"metadata"`
}

func NewTenderIQHandler(geminiService *GeminiService, vectorStore *VectorStore, pdfParser *PDFParser) *TenderIQHandler {
	return &TenderIQHandler{
		geminiService: geminiService,
		vectorStore:   vectorStore,
		pdfParser:     pdfParser,
	}
}

// Upload and parse PDF document
func (h *TenderIQHandler) UploadDocument(c echo.Context) error {
	// Get the uploaded file
	file, err := c.FormFile("file")
	if err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "No file uploaded or invalid file",
		})
	}

	// Check if it's a PDF file
	if !strings.HasSuffix(strings.ToLower(file.Filename), ".pdf") {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Only PDF files are supported",
		})
	}

	// Open the uploaded file
	src, err := file.Open()
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to open uploaded file",
		})
	}
	defer src.Close()

	// Read file content into memory
	fileContent, err := io.ReadAll(src)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to read file content",
		})
	}

	// Create a reader from the content
	reader := bytes.NewReader(fileContent)

	// Extract text from PDF
	extractedText, err := h.pdfParser.ExtractText(reader, int64(len(fileContent)))
	if err != nil {
		log.Printf("PDF parsing error: %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to extract text from PDF: " + err.Error(),
		})
	}

	// Extract metadata
	reader.Seek(0, 0) // Reset reader position
	metadata, err := h.pdfParser.ExtractMetadata(reader, int64(len(fileContent)))
	if err != nil {
		log.Printf("Metadata extraction error: %v", err)
		metadata = make(map[string]interface{})
	}

	// Add filename to metadata
	metadata["filename"] = file.Filename
	metadata["file_size"] = len(fileContent)

	// Store document in vector store
	docID, err := h.vectorStore.AddDocument(extractedText, metadata)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to store document: " + err.Error(),
		})
	}

	pages, _ := metadata["num_pages"].(int)
	
	response := UploadResponse{
		DocumentID: docID,
		Filename:   file.Filename,
		Pages:      pages,
		Metadata:   metadata,
		Message:    "Document uploaded and processed successfully",
	}

	return c.JSON(http.StatusOK, response)
}

// Analyze document with Gemini AI
func (h *TenderIQHandler) AnalyzeDocument(c echo.Context) error {
	var req AnalysisRequest
	if err := c.Bind(&req); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Invalid request format",
		})
	}

	if req.DocumentID == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Document ID is required",
		})
	}

	if req.Query == "" {
		req.Query = "Provide a comprehensive analysis of this tender document including key requirements, financial details, and important dates."
	}

	// Get document from vector store
	document, exists := h.vectorStore.GetDocument(req.DocumentID)
	if !exists {
		return c.JSON(http.StatusNotFound, map[string]string{
			"error": "Document not found",
		})
	}

	// Search for relevant chunks
	relevantChunks, err := h.vectorStore.SearchSimilar(req.Query, 5)
	if err != nil {
		log.Printf("Vector search error: %v", err)
		relevantChunks = []SearchResult{}
	}

	// Combine relevant chunks for context
	var contextText strings.Builder
	contextText.WriteString("DOCUMENT SUMMARY:\n")
	contextText.WriteString(document.Content[:min(2000, len(document.Content))])
	contextText.WriteString("\n\nRELEVANT SECTIONS:\n")
	
	for _, chunk := range relevantChunks {
		contextText.WriteString(fmt.Sprintf("- %s\n", chunk.Content))
	}

	// Analyze with Gemini
	analysisJSON, err := h.geminiService.AnalyzeTenderDocument(contextText.String(), req.Query)
	if err != nil {
		log.Printf("Gemini analysis error: %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to analyze document: " + err.Error(),
		})
	}

	// Parse the JSON response from Gemini
	var tenderAnalysis TenderAnalysis
	if err := json.Unmarshal([]byte(analysisJSON), &tenderAnalysis); err != nil {
		log.Printf("JSON parsing error: %v", err)
		// Fallback to a default structure if JSON parsing fails
		tenderAnalysis = TenderAnalysis{
			TenderID:         "Not extracted",
			Title:            "Document Analysis",
			DueDate:          "Not specified",
			IssuingAuthority: "Not specified",
			ContractValue:    "Not specified",
			ProjectOverview:  analysisJSON, // Use raw text as fallback
			FinancialReqs: FinancialRequirements{
				ContractValue: "Not specified",
				EMD:          "Not specified",
				PerformanceBG: "Not specified",
				DocumentFees:  "Not specified",
			},
			EligibilityHighlights: []string{"Analysis available in project overview"},
			ImportantDates: ImportantDates{
				PreBidQueries:       "Not specified",
				BidSubmission:       "Not specified",
				TechnicalBidOpening: "Not specified",
				FinancialBidOpening: "Not specified",
			},
			RiskAnalysis: RiskAnalysis{
				PenaltyRisk: "Not specified",
				Retention:   "Not specified",
				KeyRisks:    []string{"Please review document manually"},
			},
		}
	}

	response := AnalysisResponse{
		DocumentID:     req.DocumentID,
		Query:          req.Query,
		Analysis:       tenderAnalysis,
		RelevantChunks: relevantChunks,
		Message:        "Document analysis completed successfully",
	}

	return c.JSON(http.StatusOK, response)
}

// List all uploaded documents
func (h *TenderIQHandler) ListDocuments(c echo.Context) error {
	docIDs := h.vectorStore.ListDocuments()
	var documents []DocumentInfo

	for _, docID := range docIDs {
		if doc, exists := h.vectorStore.GetDocument(docID); exists {
			filename, _ := doc.Metadata["filename"].(string)
			pages, _ := doc.Metadata["num_pages"].(int)
			
			documents = append(documents, DocumentInfo{
				ID:       docID,
				Filename: filename,
				Pages:    pages,
				Metadata: doc.Metadata,
			})
		}
	}

	response := DocumentListResponse{
		Documents: documents,
	}

	return c.JSON(http.StatusOK, response)
}

// Delete a document
func (h *TenderIQHandler) DeleteDocument(c echo.Context) error {
	docID := c.Param("id")
	if docID == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Document ID is required",
		})
	}

	deleted := h.vectorStore.DeleteDocument(docID)
	if !deleted {
		return c.JSON(http.StatusNotFound, map[string]string{
			"error": "Document not found",
		})
	}

	return c.JSON(http.StatusOK, map[string]string{
		"message": "Document deleted successfully",
	})
}

// Get document details
func (h *TenderIQHandler) GetDocument(c echo.Context) error {
	docID := c.Param("id")
	if docID == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Document ID is required",
		})
	}

	document, exists := h.vectorStore.GetDocument(docID)
	if !exists {
		return c.JSON(http.StatusNotFound, map[string]string{
			"error": "Document not found",
		})
	}

	return c.JSON(http.StatusOK, document)
}

// Search within documents
func (h *TenderIQHandler) SearchDocuments(c echo.Context) error {
	query := c.QueryParam("q")
	if query == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Search query is required",
		})
	}

	results, err := h.vectorStore.SearchSimilar(query, 10)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Search failed: " + err.Error(),
		})
	}

	return c.JSON(http.StatusOK, map[string]interface{}{
		"query":   query,
		"results": results,
	})
}

// AnalyzeSections performs section-wise analysis of a document
func (h *TenderIQHandler) AnalyzeSections(c echo.Context) error {
	var request struct {
		DocumentID string `json:"document_id"`
	}

	if err := c.Bind(&request); err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Invalid request format",
		})
	}

	if request.DocumentID == "" {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Document ID is required",
		})
	}

	// Get document from vector store
	doc, exists := h.vectorStore.GetDocument(request.DocumentID)
	if !exists {
		return c.JSON(http.StatusNotFound, map[string]string{
			"error": "Document not found",
		})
	}

	// Perform section-wise analysis using Gemini
	result, err := h.geminiService.ExtractSectionwiseAnalysis(doc.Content)
	if err != nil {
		log.Printf("Section-wise analysis error: %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to analyze document sections",
		})
	}

	return c.JSON(http.StatusOK, result)
}

// Helper function for min
func minValue(a, b int) int {
	if a < b {
		return a
	}
	return b
}
