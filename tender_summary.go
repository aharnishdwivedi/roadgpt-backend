package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/google/generative-ai-go/genai"
	"github.com/labstack/echo/v4"
)

// TenderSummaryData represents the one-pager tender summary structure
type TenderSummaryData struct {
	ProjectOverview       string                      `json:"project_overview"`
	EligibilityHighlights []string                    `json:"eligibility_highlights"`
	ImportantDates        TenderSummaryImportantDates `json:"important_dates"`
	FinancialRequirements TenderSummaryFinancialReqs  `json:"financial_requirements"`
	RiskAnalysis          TenderSummaryRiskAnalysis   `json:"risk_analysis"`
}

type TenderSummaryImportantDates struct {
	PreBidQueries string      `json:"pre_bid_queries"`
	BidSubmission string      `json:"bid_submission"`
	OtherDates    []DateEntry `json:"other_dates"`
}

type DateEntry struct {
	Name string `json:"name"`
	Date string `json:"date"`
}

type TenderSummaryFinancialReqs struct {
	ContractValue string `json:"contract_value"`
	DocumentFees  string `json:"document_fees"`
}

type TenderSummaryRiskAnalysis struct {
	PenaltyRisk string      `json:"penalty_risk"`
	OtherRisks  []RiskEntry `json:"other_risks"`
}

type RiskEntry struct {
	Name   string `json:"name"`
	Detail string `json:"detail"`
}

type Chunk struct {
	StartPage int
	EndPage   int
	Text      string
}

type TenderSummaryResult struct {
	Mode          string            `json:"mode"`
	Final         TenderSummaryData `json:"final"`
	RawSingle     string            `json:"raw_single,omitempty"`
	PartialsCount int               `json:"partials_count,omitempty"`
}

// TenderSummaryExtractor handles tender summary extraction
type TenderSummaryExtractor struct {
	geminiService *GeminiService
	pdfParser     *PDFParser
}

// NewTenderSummaryExtractor creates a new tender summary extractor
func NewTenderSummaryExtractor(geminiService *GeminiService, pdfParser *PDFParser) *TenderSummaryExtractor {
	return &TenderSummaryExtractor{
		geminiService: geminiService,
		pdfParser:     pdfParser,
	}
}

// Prompt templates for tender summary
const TENDER_SUMMARY_SINGLE_DOC_PROMPT = `You are an expert legal/tender document parser. From the DOCUMENT extract a Tender Summary (One Pager) as ONE strict JSON object with the exact keys:

{
  "project_overview": "short paragraph summarizing the project (do NOT invent)",
  "eligibility_highlights": [ up to 4 most relevant eligibility items (strings) ],
  "important_dates": {
     "pre_bid_queries": "date or date range or text (if present)",
     "bid_submission": "date/time",
     "other_dates": [ {"name": "...", "date": "..."} ]
  },
  "financial_requirements": {
     "contract_value": "value (one token or short text, e.g., 'INR 10,00,00,000')",
     "document_fees": "value (one token or short text)"
  },
  "risk_analysis": {
     "penalty_risk": "concise description of penalty risk if present (one sentence)",
     "other_risks": [ {"name":"...", "detail":"..."} ]
  }
}

Rules:
- RESPOND IN VALID JSON ONLY, and nothing else.
- Do NOT invent facts; if a field is not present, return an empty string or empty list as appropriate.
- For every extracted value, include page provenance when possible by appending " (page X)" or " (pages X-Y)" inside the string.
- For eligibility_highlights select the up to 4 most important/representative items from the document (prioritize clear bullet items or eligibility criteria).
- For important_dates, try to find Pre-bid Queries and Bid Submission dates explicitly; place other notable dates in other_dates array.
- For financial_requirements, return the short token/value for Contract Value and Document Fees. If multiple values exist, prefer the one clearly labelled 'Contract Value' and the published tender value.
- For penalty_risk, summarize any clause that describes penalties or liquidated damages in one sentence.

DOCUMENT:
<<<DOC>>>`

const TENDER_SUMMARY_CHUNK_PROMPT = `You are an expert tender document parser. From the DOCUMENT CHUNK extract the same Tender Summary object (use schema described below) and return a single VALID JSON object.

Schema (exact keys):
{
  "project_overview": "...",
  "eligibility_highlights": [...],
  "important_dates": {"pre_bid_queries":"...","bid_submission":"...","other_dates":[{"name":"...","date":"..."}]},
  "financial_requirements": {"contract_value":"...","document_fees":"..."},
  "risk_analysis": {"penalty_risk":"...","other_risks":[{"name":"...","detail":"..."}]}
}

Rules:
- Return JSON ONLY.
- Include page provenance (append page numbers in parentheses).
DOCUMENT CHUNK:
<<<DOC>>>`

// ExtractTenderSummary performs tender summary extraction with single-call and chunked fallback
func (tse *TenderSummaryExtractor) ExtractTenderSummary(pdfPath string) (*TenderSummaryResult, error) {
	log.Printf("Starting tender summary extraction for: %s", pdfPath)

	// Extract pages from PDF
	file, err := os.Open(pdfPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF: %v", err)
	}
	defer file.Close()

	pages, err := tse.pdfParser.ExtractTextByPage(file)
	if err != nil {
		return nil, fmt.Errorf("failed to extract pages: %v", err)
	}

	log.Printf("Extracted %d pages from PDF", len(pages))

	// Prepare full document text
	var fullTextBuilder strings.Builder
	for i, page := range pages {
		fullTextBuilder.WriteString(fmt.Sprintf("[PAGE:%d]\n%s\n\n", i+1, page))
	}
	fullText := fullTextBuilder.String()

	// 1. Single-call attempt with gemini-2.5-flash
	log.Println("=== Attempting single full-document extraction with gemini-2.5-flash ===")
	singlePrompt := strings.Replace(TENDER_SUMMARY_SINGLE_DOC_PROMPT, "<<<DOC>>>", fullText, 1)

	singleResp, err := tse.callGeminiFlash(singlePrompt)
	if err != nil {
		log.Printf("Single-call error: %v", err)
	} else {
		log.Printf("Single-call RAW preview: %s", truncateString(singleResp, 2000))

		parsed := tse.safeParseJSON(singleResp)
		if summaryData, ok := parsed.(*TenderSummaryData); ok && summaryData != nil {
			log.Println("Single-call parsed OK — returning result")
			return &TenderSummaryResult{
				Mode:      "single_call",
				Final:     *summaryData,
				RawSingle: singleResp,
			}, nil
		}
		log.Println("Single-call returned unparsable structure — falling back to chunked extraction")
	}

	// 2. Fallback: chunked extraction
	log.Println("=== Running chunked extraction (fallback) with gemini-2.5-flash ===")
	chunks := tse.makeChunksFromPages(pages, 6, 1)
	log.Printf("Built %d chunk(s)", len(chunks))

	var partialObjs []TenderSummaryData
	for i, chunk := range chunks {
		log.Printf("--- chunk %d/%d pages %d-%d ---", i+1, len(chunks), chunk.StartPage, chunk.EndPage)

		chunkPrompt := strings.Replace(TENDER_SUMMARY_CHUNK_PROMPT, "<<<DOC>>>", chunk.Text, 1)
		resp, err := tse.callGeminiFlash(chunkPrompt)
		if err != nil {
			log.Printf("Chunk %d error: %v", i+1, err)
			partialObjs = append(partialObjs, tse.getEmptyTenderSummary())
			continue
		}

		log.Printf("RAW preview: %s", truncateString(resp, 2000))

		parsed := tse.safeParseJSON(resp)
		if summaryData, ok := parsed.(*TenderSummaryData); ok && summaryData != nil {
			// Add provenance to project overview if missing
			if summaryData.ProjectOverview != "" && !strings.Contains(strings.ToLower(summaryData.ProjectOverview), "page") {
				summaryData.ProjectOverview = fmt.Sprintf("%s (pages %d-%d)", summaryData.ProjectOverview, chunk.StartPage, chunk.EndPage)
			}

			// Add provenance to dates if missing
			tse.addProvenanceToSummary(summaryData, chunk.StartPage, chunk.EndPage)

			partialObjs = append(partialObjs, *summaryData)
		} else {
			log.Printf("Warning: chunk %d parsing failed; storing empty placeholder", i+1)
			partialObjs = append(partialObjs, tse.getEmptyTenderSummary())
		}

		// Throttle requests
		time.Sleep(500 * time.Millisecond)
	}

	// 3. Aggregate results
	log.Println("=== Aggregating partial results ===")
	final := tse.mergeTenderObjects(partialObjs)

	return &TenderSummaryResult{
		Mode:          "chunked_fallback",
		Final:         final,
		PartialsCount: len(partialObjs),
	}, nil
}

// callGeminiFlash calls Gemini Flash model
func (tse *TenderSummaryExtractor) callGeminiFlash(prompt string) (string, error) {
	if tse.geminiService == nil || tse.geminiService.flashModel == nil {
		return "", fmt.Errorf("gemini service not initialized")
	}

	ctx := context.Background()
	resp, err := tse.geminiService.flashModel.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return "", err
	}

	if resp.Candidates == nil || len(resp.Candidates) == 0 {
		return "", fmt.Errorf("no candidates in response")
	}

	candidate := resp.Candidates[0]
	if candidate.Content == nil || len(candidate.Content.Parts) == 0 {
		return "", fmt.Errorf("no content in candidate")
	}

	var result strings.Builder
	for _, part := range candidate.Content.Parts {
		result.WriteString(fmt.Sprintf("%v", part))
	}

	return result.String(), nil
}

// makeChunksFromPages creates chunks from pages
func (tse *TenderSummaryExtractor) makeChunksFromPages(pages []string, pagesPerChunk, overlapPages int) []Chunk {
	var chunks []Chunk
	n := len(pages)
	if n == 0 {
		return chunks
	}

	i := 0
	for i < n {
		start := i
		end := min(n, i+pagesPerChunk)

		var textBuilder strings.Builder
		for idx := start; idx < end; idx++ {
			textBuilder.WriteString(fmt.Sprintf("[PAGE:%d]\n%s\n\n", idx+1, pages[idx]))
		}

		chunks = append(chunks, Chunk{
			StartPage: start + 1,
			EndPage:   end,
			Text:      textBuilder.String(),
		})

		if end == n {
			break
		}
		i = end - overlapPages
	}

	return chunks
}

// safeParseJSON safely parses JSON with fallback strategies
func (tse *TenderSummaryExtractor) safeParseJSON(raw string) interface{} {
	if raw == "" {
		return nil
	}

	cleaned := tse.cleanModelOutput(raw)

	// Try direct parsing
	var summaryData TenderSummaryData
	if err := json.Unmarshal([]byte(cleaned), &summaryData); err == nil {
		return &summaryData
	}

	// Try extracting JSON block
	block := tse.extractJSONBlock(cleaned)
	if block != "" {
		if err := json.Unmarshal([]byte(block), &summaryData); err == nil {
			return &summaryData
		}

		// Progressive trim fallback
		for end := len(block); end > 0; end-- {
			if err := json.Unmarshal([]byte(block[:end]), &summaryData); err == nil {
				return &summaryData
			}
		}
	}

	return nil
}

// cleanModelOutput cleans model output
func (tse *TenderSummaryExtractor) cleanModelOutput(raw string) string {
	if raw == "" {
		return ""
	}

	// Remove code block markers
	re1 := regexp.MustCompile("(?i)```(?:json)?\\s*")
	re2 := regexp.MustCompile("\\s*```")

	cleaned := re1.ReplaceAllString(raw, "")
	cleaned = re2.ReplaceAllString(cleaned, "")

	return strings.TrimSpace(cleaned)
}

// extractJSONBlock extracts JSON block from text
func (tse *TenderSummaryExtractor) extractJSONBlock(text string) string {
	if text == "" {
		return ""
	}

	// Find JSON objects or arrays
	re := regexp.MustCompile(`(\{[\s\S]*\}|\[[\s\S]*\])`)
	matches := re.FindAllString(text, -1)

	if len(matches) == 0 {
		return ""
	}

	// Return the longest match
	longest := ""
	for _, match := range matches {
		if len(match) > len(longest) {
			longest = match
		}
	}

	return longest
}

// getEmptyTenderSummary returns empty tender summary structure
func (tse *TenderSummaryExtractor) getEmptyTenderSummary() TenderSummaryData {
	return TenderSummaryData{
		ProjectOverview:       "",
		EligibilityHighlights: []string{},
		ImportantDates: TenderSummaryImportantDates{
			PreBidQueries: "",
			BidSubmission: "",
			OtherDates:    []DateEntry{},
		},
		FinancialRequirements: TenderSummaryFinancialReqs{
			ContractValue: "",
			DocumentFees:  "",
		},
		RiskAnalysis: TenderSummaryRiskAnalysis{
			PenaltyRisk: "",
			OtherRisks:  []RiskEntry{},
		},
	}
}

// addProvenanceToSummary adds page provenance to summary fields
func (tse *TenderSummaryExtractor) addProvenanceToSummary(summary *TenderSummaryData, startPage, endPage int) {
	pageRange := fmt.Sprintf("(pages %d-%d)", startPage, endPage)

	// Add to dates if missing provenance
	if summary.ImportantDates.PreBidQueries != "" && !strings.Contains(strings.ToLower(summary.ImportantDates.PreBidQueries), "page") {
		summary.ImportantDates.PreBidQueries = fmt.Sprintf("%s %s", summary.ImportantDates.PreBidQueries, pageRange)
	}

	if summary.ImportantDates.BidSubmission != "" && !strings.Contains(strings.ToLower(summary.ImportantDates.BidSubmission), "page") {
		summary.ImportantDates.BidSubmission = fmt.Sprintf("%s %s", summary.ImportantDates.BidSubmission, pageRange)
	}

	for i := range summary.ImportantDates.OtherDates {
		if summary.ImportantDates.OtherDates[i].Date != "" && !strings.Contains(strings.ToLower(summary.ImportantDates.OtherDates[i].Date), "page") {
			summary.ImportantDates.OtherDates[i].Date = fmt.Sprintf("%s %s", summary.ImportantDates.OtherDates[i].Date, pageRange)
		}
	}
}

// mergeTenderObjects merges multiple tender summary objects
func (tse *TenderSummaryExtractor) mergeTenderObjects(objs []TenderSummaryData) TenderSummaryData {
	final := tse.getEmptyTenderSummary()

	// Project overview: choose longest non-empty
	bestProj := ""
	for _, obj := range objs {
		po := strings.TrimSpace(obj.ProjectOverview)
		if po != "" && len(po) > len(bestProj) {
			bestProj = po
		}
	}
	final.ProjectOverview = bestProj

	// Eligibility highlights: collect and dedupe, take up to 4
	var candElig []string
	for _, obj := range objs {
		for _, item := range obj.EligibilityHighlights {
			if strings.TrimSpace(item) != "" {
				candElig = append(candElig, strings.TrimSpace(item))
			}
		}
	}

	// Dedupe preserving order
	seen := make(map[string]bool)
	var dedup []string
	for _, item := range candElig {
		key := strings.ToLower(item)
		if !seen[key] {
			dedup = append(dedup, item)
			seen[key] = true
		}
	}

	if len(dedup) > 4 {
		dedup = dedup[:4]
	}
	final.EligibilityHighlights = dedup

	// Important dates: prefer non-empty fields
	for _, obj := range objs {
		if obj.ImportantDates.PreBidQueries != "" && final.ImportantDates.PreBidQueries == "" {
			final.ImportantDates.PreBidQueries = obj.ImportantDates.PreBidQueries
		}
		if obj.ImportantDates.BidSubmission != "" && final.ImportantDates.BidSubmission == "" {
			final.ImportantDates.BidSubmission = obj.ImportantDates.BidSubmission
		}
		for _, other := range obj.ImportantDates.OtherDates {
			final.ImportantDates.OtherDates = append(final.ImportantDates.OtherDates, other)
		}
	}

	// Financial requirements
	for _, obj := range objs {
		if obj.FinancialRequirements.ContractValue != "" && final.FinancialRequirements.ContractValue == "" {
			final.FinancialRequirements.ContractValue = obj.FinancialRequirements.ContractValue
		}
		if obj.FinancialRequirements.DocumentFees != "" && final.FinancialRequirements.DocumentFees == "" {
			final.FinancialRequirements.DocumentFees = obj.FinancialRequirements.DocumentFees
		}
	}

	// Risk analysis
	for _, obj := range objs {
		if obj.RiskAnalysis.PenaltyRisk != "" && final.RiskAnalysis.PenaltyRisk == "" {
			final.RiskAnalysis.PenaltyRisk = obj.RiskAnalysis.PenaltyRisk
		}
		for _, risk := range obj.RiskAnalysis.OtherRisks {
			final.RiskAnalysis.OtherRisks = append(final.RiskAnalysis.OtherRisks, risk)
		}
	}

	// Dedupe other dates
	final.ImportantDates.OtherDates = tse.dedupeOtherDates(final.ImportantDates.OtherDates)

	// Dedupe other risks
	final.RiskAnalysis.OtherRisks = tse.dedupeOtherRisks(final.RiskAnalysis.OtherRisks)

	return final
}

// dedupeOtherDates removes duplicate date entries
func (tse *TenderSummaryExtractor) dedupeOtherDates(dates []DateEntry) []DateEntry {
	seen := make(map[string]bool)
	var unique []DateEntry

	for _, date := range dates {
		key := strings.ToLower(fmt.Sprintf("%s|%s", date.Name, date.Date))
		if !seen[key] {
			unique = append(unique, date)
			seen[key] = true
		}
	}

	return unique
}

// dedupeOtherRisks removes duplicate risk entries
func (tse *TenderSummaryExtractor) dedupeOtherRisks(risks []RiskEntry) []RiskEntry {
	seen := make(map[string]bool)
	var unique []RiskEntry

	for _, risk := range risks {
		key := strings.ToLower(fmt.Sprintf("%s|%s", risk.Name, risk.Detail))
		if !seen[key] {
			unique = append(unique, risk)
			seen[key] = true
		}
	}

	return unique
}

// HTTP handler for tender summary extraction
func (tse *TenderSummaryExtractor) HandleTenderSummaryExtraction(c echo.Context) error {
	// Parse multipart form
	err := c.Request().ParseMultipartForm(32 << 20) // 32MB max
	if err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "Failed to parse form"})
	}

	file, header, err := c.Request().FormFile("pdf")
	if err != nil {
		return c.JSON(http.StatusBadRequest, map[string]string{"error": "No PDF file provided"})
	}
	defer file.Close()

	// Save uploaded file temporarily
	tempPath, cleanup, err := saveUploadedFile(file, header.Filename)
	if err != nil {
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": "Failed to save uploaded file"})
	}
	defer cleanup()

	log.Printf("Processing tender summary extraction for: %s", header.Filename)

	// Extract tender summary
	result, err := tse.ExtractTenderSummary(tempPath)
	if err != nil {
		log.Printf("Tender summary extraction failed: %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{"error": fmt.Sprintf("Extraction failed: %v", err)})
	}

	log.Printf("Tender summary extraction completed successfully for: %s", header.Filename)
	return c.JSON(http.StatusOK, result)
}

// Helper function to truncate strings
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// Helper function for min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// saveUploadedFile saves an uploaded file temporarily and returns the path and cleanup function
func saveUploadedFile(file multipart.File, filename string) (string, func(), error) {
	// Create temporary file
	tempFile, err := os.CreateTemp("", "tender_*_"+filepath.Base(filename))
	if err != nil {
		return "", nil, err
	}
	
	tempPath := tempFile.Name()
	
	// Copy uploaded file to temporary file
	_, err = io.Copy(tempFile, file)
	tempFile.Close()
	
	if err != nil {
		os.Remove(tempPath)
		return "", nil, err
	}
	
	// Return path and cleanup function
	cleanup := func() {
		os.Remove(tempPath)
	}
	
	return tempPath, cleanup, nil
}
