package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"regexp"
	"strings"
	"time"

	"github.com/google/generative-ai-go/genai"
	"github.com/labstack/echo/v4"
	"google.golang.org/api/option"
)

// Scope of Work data structures
type ProjectOverview struct {
	ProjectName      string `json:"project_name"`
	Location         string `json:"location"`
	TotalLength      string `json:"total_length"`
	ProjectDuration  string `json:"project_duration"`
	ContractValue    string `json:"contract_value"`
}

type MajorWorkComponent struct {
	SNo                   string `json:"s_no"`
	WorkDescription       string `json:"work_description"`
	QuantitySpecification string `json:"quantity_specification"`
	Unit                  string `json:"unit"`
}

type TechnicalStandard struct {
	Component             string `json:"component"`
	StandardSpecification string `json:"standard_specification"`
	ComplianceRequired    string `json:"compliance_required"`
}

type ScopeOfWorkData struct {
	ProjectOverview       ProjectOverview        `json:"project_overview"`
	MajorWorkComponents   []MajorWorkComponent   `json:"major_work_components"`
	TechnicalStandards    []TechnicalStandard    `json:"technical_standards"`
}

type SOWExtractionResult struct {
	Mode                string              `json:"mode"`
	Final               ScopeOfWorkData     `json:"final"`
	ChunkParsedList     []ScopeOfWorkData   `json:"chunk_parsed_list,omitempty"`
	RawSingle           string              `json:"raw_single,omitempty"`
	Error               string              `json:"error,omitempty"`
}

// Prompts
const SINGLE_CALL_PROMPT = `You are an expert document parser. Extract ONLY the following three structured fields from the DOCUMENT (do not invent anything):

1) project_overview: JSON object with keys:
   - project_name
   - location
   - total_length
   - project_duration
   - contract_value

   If a field is not present, set it to an empty string.

2) major_work_components: array of objects:
   [
     { "s_no":"...", "work_description":"...", "quantity_specification":"...", "unit":"..." },
     ...
   ]
   If none present, return an empty list.

3) technical_standards: array of objects:
   [
     { "component":"...", "standard_specification":"...", "compliance_required":"..." },
     ...
   ]
   If none present, return an empty list.

IMPORTANT:
• RESPOND IN VALID JSON ONLY with EXACT KEYS: {"project_overview": {...}, "major_work_components": [...], "technical_standards": [...]}
• Do NOT add explanations, do NOT include any text outside the single JSON object.
• Include brief page references where you can (e.g., "(page 4)") in values when the source is clear.

DOCUMENT:
<<<DOC>>>`

const CHUNK_EXTRACTION_PROMPT = `You are a document parser. From the provided DOCUMENT CHUNK extract ONLY the three structured fields as JSON:
• project_overview: {project_name, location, total_length, project_duration, contract_value}
• major_work_components: [{"s_no","work_description","quantity_specification","unit"}]
• technical_standards: [{"component","standard_specification","compliance_required"}]

Return a single VALID JSON object only.

DOCUMENT CHUNK:
<<<DOC>>>`

const AGGREGATION_PROMPT = `You are given multiple JSON extraction results (chunk-level). Combine them into a single consolidated JSON with schema:
{
  "project_overview": { "project_name":"...", "location":"...", "total_length":"...", "project_duration":"...", "contract_value":"..." },
  "major_work_components": [ ... ],
  "technical_standards": [ ... ]
}

Rules:
• Prefer non-empty values for project_overview; if multiple conflicting non-empty values exist prefer page-referenced values or the value that appears most frequently.
• Merge lists and deduplicate exact duplicates (case-insensitive).
• Do not invent values not present in chunk results.

Chunk findings:
<<<CHUNKS_JSON>>>`

type SOWExtractor struct {
	geminiService *GeminiService
	apiKey        string
}

func NewSOWExtractor(geminiService *GeminiService, apiKey string) *SOWExtractor {
	return &SOWExtractor{
		geminiService: geminiService,
		apiKey:        apiKey,
	}
}

// Clean and parse JSON response
func (s *SOWExtractor) cleanModelOutput(raw string) string {
	if raw == "" {
		return ""
	}
	// Remove markdown code blocks
	re := regexp.MustCompile("```(?:json)?\\s*")
	cleaned := re.ReplaceAllString(raw, "")
	cleaned = strings.ReplaceAll(cleaned, "```", "")
	return strings.TrimSpace(cleaned)
}

func (s *SOWExtractor) extractJSONBlock(text string) string {
	if text == "" {
		return ""
	}
	// Find JSON objects
	re := regexp.MustCompile(`(\{[\s\S]*\})`)
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

func (s *SOWExtractor) safeParseJSON(raw string) (*ScopeOfWorkData, error) {
	if raw == "" {
		return nil, fmt.Errorf("empty response")
	}
	
	cleaned := s.cleanModelOutput(raw)
	
	// Try direct parsing first
	var result ScopeOfWorkData
	if err := json.Unmarshal([]byte(cleaned), &result); err == nil {
		return &result, nil
	}
	
	// Extract JSON block and try again
	block := s.extractJSONBlock(cleaned)
	if block != "" {
		if err := json.Unmarshal([]byte(block), &result); err == nil {
			return &result, nil
		}
	}
	
	return nil, fmt.Errorf("failed to parse JSON response")
}

// Call Gemini model
func (s *SOWExtractor) callModelForPrompt(ctx context.Context, prompt string, modelName string) (*ScopeOfWorkData, string, error) {
	client, err := genai.NewClient(ctx, option.WithAPIKey(s.apiKey))
	if err != nil {
		return nil, "", fmt.Errorf("failed to create Gemini client: %v", err)
	}
	defer client.Close()

	model := client.GenerativeModel(modelName)
	
	// Configure model parameters
	model.SetTemperature(0.1)
	model.SetTopP(0.8)
	model.SetTopK(40)
	model.SetMaxOutputTokens(8192)

	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return nil, "", fmt.Errorf("model generation failed: %v", err)
	}

	if len(resp.Candidates) == 0 {
		return nil, "", fmt.Errorf("no response candidates")
	}

	var rawText strings.Builder
	for _, part := range resp.Candidates[0].Content.Parts {
		if txt, ok := part.(genai.Text); ok {
			rawText.WriteString(string(txt))
		}
	}

	raw := rawText.String()
	parsed, err := s.safeParseJSON(raw)
	
	return parsed, raw, err
}

// Chunk-based extraction
func (s *SOWExtractor) makeChunksFromPages(pages []string, pagesPerChunk int, overlapPages int) []map[string]interface{} {
	var chunks []map[string]interface{}
	n := len(pages)
	if n == 0 {
		return chunks
	}

	i := 0
	for i < n {
		start := i
		end := i + pagesPerChunk
		if end > n {
			end = n
		}

		var textParts []string
		for idx := start; idx < end; idx++ {
			textParts = append(textParts, fmt.Sprintf("[PAGE:%d]\n%s", idx+1, pages[idx]))
		}

		chunk := map[string]interface{}{
			"start_page": start + 1,
			"end_page":   end,
			"text":       strings.Join(textParts, "\n\n"),
		}
		chunks = append(chunks, chunk)

		if end == n {
			break
		}
		i = end - overlapPages
	}

	return chunks
}

// Programmatic merge for fallback
func (s *SOWExtractor) programmaticMerge(chunkResults []ScopeOfWorkData) ScopeOfWorkData {
	final := ScopeOfWorkData{
		ProjectOverview: ProjectOverview{},
		MajorWorkComponents: []MajorWorkComponent{},
		TechnicalStandards: []TechnicalStandard{},
	}

	// Merge project overview - take first non-empty values
	for _, chunk := range chunkResults {
		po := chunk.ProjectOverview
		if final.ProjectOverview.ProjectName == "" && po.ProjectName != "" {
			final.ProjectOverview.ProjectName = po.ProjectName
		}
		if final.ProjectOverview.Location == "" && po.Location != "" {
			final.ProjectOverview.Location = po.Location
		}
		if final.ProjectOverview.TotalLength == "" && po.TotalLength != "" {
			final.ProjectOverview.TotalLength = po.TotalLength
		}
		if final.ProjectOverview.ProjectDuration == "" && po.ProjectDuration != "" {
			final.ProjectOverview.ProjectDuration = po.ProjectDuration
		}
		if final.ProjectOverview.ContractValue == "" && po.ContractValue != "" {
			final.ProjectOverview.ContractValue = po.ContractValue
		}
	}

	// Merge and deduplicate lists
	seenMW := make(map[string]bool)
	for _, chunk := range chunkResults {
		for _, item := range chunk.MajorWorkComponents {
			key := fmt.Sprintf("%s|%s|%s|%s", item.SNo, item.WorkDescription, item.QuantitySpecification, item.Unit)
			if !seenMW[strings.ToLower(key)] {
				seenMW[strings.ToLower(key)] = true
				final.MajorWorkComponents = append(final.MajorWorkComponents, item)
			}
		}
	}

	seenTS := make(map[string]bool)
	for _, chunk := range chunkResults {
		for _, item := range chunk.TechnicalStandards {
			key := fmt.Sprintf("%s|%s|%s", item.Component, item.StandardSpecification, item.ComplianceRequired)
			if !seenTS[strings.ToLower(key)] {
				seenTS[strings.ToLower(key)] = true
				final.TechnicalStandards = append(final.TechnicalStandards, item)
			}
		}
	}

	return final
}

// Main extraction function with fallback
func (s *SOWExtractor) ExtractSOW(ctx context.Context, pages []string) (*SOWExtractionResult, error) {
	if len(pages) == 0 {
		return nil, fmt.Errorf("no pages provided")
	}

	log.Printf("Starting SOW extraction for %d pages", len(pages))

	// Prepare full document text
	var fullTextParts []string
	for i, page := range pages {
		fullTextParts = append(fullTextParts, fmt.Sprintf("[PAGE:%d]\n%s", i+1, page))
	}
	fullText := strings.Join(fullTextParts, "\n\n")

	// 1. Try single-call with gemini-2.5-pro
	log.Println("Attempting single-call extraction with gemini-2.5-pro")
	singlePrompt := strings.ReplaceAll(SINGLE_CALL_PROMPT, "<<<DOC>>>", fullText)
	
	parsed, rawSingle, err := s.callModelForPrompt(ctx, singlePrompt, "gemini-2.5-pro")
	if err == nil && parsed != nil {
		log.Println("Single-call extraction successful")
		return &SOWExtractionResult{
			Mode:      "single_call",
			Final:     *parsed,
			RawSingle: rawSingle,
		}, nil
	}

	log.Printf("Single-call failed (%v), falling back to chunked extraction", err)

	// 2. Fallback: chunked extraction with gemini-2.5-flash
	log.Println("Running chunked extraction with gemini-2.5-flash")
	chunks := s.makeChunksFromPages(pages, 6, 1)
	log.Printf("Created %d chunks", len(chunks))

	var chunkResults []ScopeOfWorkData
	for i, chunk := range chunks {
		log.Printf("Processing chunk %d/%d (pages %v-%v)", i+1, len(chunks), chunk["start_page"], chunk["end_page"])
		
		chunkPrompt := strings.ReplaceAll(CHUNK_EXTRACTION_PROMPT, "<<<DOC>>>", chunk["text"].(string))
		parsed, _, err := s.callModelForPrompt(ctx, chunkPrompt, "gemini-2.5-flash")
		
		if err != nil {
			log.Printf("Chunk %d extraction failed: %v", i+1, err)
			// Add empty placeholder
			chunkResults = append(chunkResults, ScopeOfWorkData{
				ProjectOverview:     ProjectOverview{},
				MajorWorkComponents: []MajorWorkComponent{},
				TechnicalStandards:  []TechnicalStandard{},
			})
		} else {
			chunkResults = append(chunkResults, *parsed)
		}

		// Throttle requests
		time.Sleep(400 * time.Millisecond)
	}

	// 3. Try model-based aggregation
	log.Println("Attempting model-based aggregation")
	chunksJSON, _ := json.Marshal(chunkResults)
	aggPrompt := strings.ReplaceAll(AGGREGATION_PROMPT, "<<<CHUNKS_JSON>>>", string(chunksJSON))
	
	aggregated, _, aggErr := s.callModelForPrompt(ctx, aggPrompt, "gemini-2.5-flash")
	if aggErr == nil && aggregated != nil {
		log.Println("Model-based aggregation successful")
		return &SOWExtractionResult{
			Mode:            "chunk_aggregate_model",
			Final:           *aggregated,
			ChunkParsedList: chunkResults,
		}, nil
	}

	log.Printf("Model aggregation failed (%v), using programmatic merge", aggErr)

	// 4. Programmatic merge fallback
	final := s.programmaticMerge(chunkResults)
	return &SOWExtractionResult{
		Mode:            "chunk_aggregate_programmatic",
		Final:           final,
		ChunkParsedList: chunkResults,
	}, nil
}

// HTTP handler for scope of work extraction
func handleScopeOfWorkExtraction(c echo.Context, sowExtractor *SOWExtractor, pdfParser *PDFParser) error {
	// Parse multipart form
	form, err := c.MultipartForm()
	if err != nil {
		log.Printf("Error parsing multipart form: %v", err)
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "Invalid multipart form data",
		})
	}

	// Get the uploaded file
	files := form.File["file"]
	if len(files) == 0 {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "No file uploaded",
		})
	}

	file := files[0]
	log.Printf("Processing scope of work extraction for file: %s", file.Filename)

	// Open the uploaded file
	src, err := file.Open()
	if err != nil {
		log.Printf("Error opening uploaded file: %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to open uploaded file",
		})
	}
	defer src.Close()

	// Extract text from PDF
	pages, err := pdfParser.ExtractTextByPage(src)
	if err != nil {
		log.Printf("Error extracting text from PDF: %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": "Failed to extract text from PDF",
		})
	}

	if len(pages) == 0 {
		return c.JSON(http.StatusBadRequest, map[string]string{
			"error": "No text content found in PDF",
		})
	}

	// Extract scope of work
	ctx := context.Background()
	result, err := sowExtractor.ExtractSOW(ctx, pages)
	if err != nil {
		log.Printf("Error extracting scope of work: %v", err)
		return c.JSON(http.StatusInternalServerError, map[string]string{
			"error": fmt.Sprintf("Failed to extract scope of work: %v", err),
		})
	}

	log.Printf("Scope of work extraction completed successfully using mode: %s", result.Mode)
	return c.JSON(http.StatusOK, result)
}
