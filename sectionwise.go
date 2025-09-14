package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"regexp"
	"strings"
	"time"

	"github.com/google/generative-ai-go/genai"
)

type SectionAnalysis struct {
	SectionName       string             `json:"section_name"`
	SectionSummary    string             `json:"section_summary"`
	KeyConsiderations []KeyConsideration `json:"key_considerations"`
}

type KeyConsideration struct {
	Consideration string  `json:"consideration"`
	IsCritical    bool    `json:"is_critical"`
	PageNumbers   []int   `json:"page_numbers,omitempty"`
}

type SectionwiseResult struct {
	Mode                  string            `json:"mode"`
	Final                 []SectionAnalysis `json:"final"`
	RawSingle             string            `json:"raw_single,omitempty"`
	ProcessedChunks       int               `json:"processed_chunks,omitempty"`
	SectionsCount         int               `json:"sections_count,omitempty"`
	CompletedSectionCount int               `json:"completed_section_count,omitempty"`
}

type OptimizedChunk struct {
	StartPage int
	EndPage   int
	Text      string
	PageRange string
}

const SINGLE_DOC_PROMPT = `You are an expert document parser. From the DOCUMENT extract every logical section and return a JSON array of section objects.

Each section object must have exactly these keys:
- "section_name": short title of the section (string)
- "section_summary": concise 1-3 sentence summary of the section content (do NOT invent; summarize only what's present)
- "key_considerations": array of objects, each with:
    { "consideration": "...", "is_critical": true|false, "page_numbers": [1, 2, 3] }

Rules:
- RESPOND WITH A SINGLE VALID JSON ARRAY (list) and nothing else.
- Mark "is_critical": true if the language explicitly flags it as critical/mandatory/penalty/limit OR if it contains strong actionable items (deadlines, penalties, percentages, "CRITICAL", "must", "shall", "mandatory").
- Include page numbers wherever you can (use page markers in the DOCUMENT).
- If you cannot find sections, return an empty list [].

DOCUMENT:
%s`

const CHUNK_PROMPT = `You are an expert document parser. From the DOCUMENT CHUNK extract all sections found in this chunk and return a single VALID JSON ARRAY of section objects with keys:
- section_name
- section_summary  
- key_considerations: [ { "consideration": "...", "is_critical": true|false, "page_numbers": [1, 2, 3] }, ... ]

Rules:
- Return JSON ONLY (no explanation).
- Use page markers in the chunk for provenance.

DOCUMENT CHUNK:
%s`

const AGGREGATE_PROMPT = `You are given multiple JSON arrays (chunk-level extractions) representing sections found across a document. Combine them into one consolidated JSON array of sections.

Rules:
- For sections with the same or very similar names, merge them into one section; keep the longest/best summary, and combine key_considerations, deduplicating exact duplicate items (case-insensitive).
- Keep page provenance for every key_consideration.
- Do NOT invent facts.
- Return a single VALID JSON ARRAY of section objects with keys: section_name, section_summary, key_considerations.

Chunk-level JSON arrays (one per chunk):
%s`

func (g *GeminiService) ExtractSectionwiseAnalysis(documentText string) (*SectionwiseResult, error) {
	if g.client == nil || g.proModel == nil || g.flashModel == nil {
		return nil, fmt.Errorf("gemini client not initialized")
	}

	ctx := context.Background()

	// 1. Attempt full-document single-call with Gemini 2.5 Pro
	log.Printf("=== Attempting single-call full-document with Gemini 2.5 Pro ===")
	log.Printf("(If this fails or returns unparsable JSON, we'll try fallback single-call then chunked extraction.)")

	prompt := fmt.Sprintf(SINGLE_DOC_PROMPT, documentText)
	resp, err := g.proModel.GenerateContent(ctx, genai.Text(prompt))

	if err == nil && len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		var result string
		for _, part := range resp.Candidates[0].Content.Parts {
			if textPart, ok := part.(genai.Text); ok {
				result += string(textPart)
			}
		}

		result = cleanJSONResponse(result)

		var sections []SectionAnalysis
		if json.Unmarshal([]byte(result), &sections) == nil && len(sections) > 0 {
			log.Printf("Primary single-call parsed as list. Returning result.")
			return &SectionwiseResult{
				Mode:      "single_primary",
				Final:     sections,
				RawSingle: result,
			}, nil
		}
		log.Printf("Primary single-call failed or returned unparsable output. Preview (truncated): %s", truncateStringForSections(result, 2000))
	} else {
		log.Printf("Primary single-call failed: %v", err)
	}

	// 2. Try single-call with Gemini 2.5 Flash fallback
	log.Printf("=== Attempting single-call full-document with fallback model Gemini 2.5 Flash ===")
	resp, err = g.flashModel.GenerateContent(ctx, genai.Text(prompt))

	if err == nil && len(resp.Candidates) > 0 && len(resp.Candidates[0].Content.Parts) > 0 {
		var result string
		for _, part := range resp.Candidates[0].Content.Parts {
			if textPart, ok := part.(genai.Text); ok {
				result += string(textPart)
			}
		}

		result = cleanJSONResponse(result)

		var sections []SectionAnalysis
		if json.Unmarshal([]byte(result), &sections) == nil && len(sections) > 0 {
			log.Printf("Secondary single-call parsed as list. Returning result.")
			return &SectionwiseResult{
				Mode:      "single_secondary",
				Final:     sections,
				RawSingle: result,
			}, nil
		}
		log.Printf("Secondary single-call failed or returned unparsable output. Preview (truncated): %s", truncateStringForSections(result, 2000))
	} else {
		log.Printf("Secondary single-call failed: %v", err)
	}

	// 3. Fallback: optimized chunked extraction using Gemini 2.5 Flash
	log.Printf("=== Falling back to optimized chunked extraction using Gemini 2.5 Flash ===")

	// Extract text by pages and create optimized chunks
	pages := g.extractTextByPage(documentText)
	log.Printf("PDF pages: %d", len(pages))

	chunks := g.makeChunksFromPages(pages, 6, 1) // 6 pages per chunk, 1 page overlap
	log.Printf("Built %d chunk(s) (pages_per_chunk=6, overlap=1)", len(chunks))

	// Prefilter chunks to only those likely containing sections
	candidateChunks := g.filterCandidateChunks(chunks)
	log.Printf("Candidate chunks to call model on (after prefilter): %d", len(candidateChunks))

	chunkResults := [][]SectionAnalysis{}
	processedCount := 0
	consecutiveNoNew := 0
	maxConsecutiveNoNew := 6
	maxRetries := 2
	processedChunks := make(map[string]bool) // Track processed chunks to avoid duplicates

	for i, chunk := range candidateChunks {
		// Skip if already processed
		chunkKey := fmt.Sprintf("%s", chunk.PageRange)
		if processedChunks[chunkKey] {
			log.Printf("Skipping already processed chunk %s", chunk.PageRange)
			continue
		}

		processedCount++
		log.Printf("--- chunk %d/%d pages %s ---", processedCount, len(candidateChunks), chunk.PageRange)

		// Mark as processed immediately to prevent retries
		processedChunks[chunkKey] = true

		// Retry logic with exponential backoff
		var chunkSections []SectionAnalysis
		success := false
		for retry := 0; retry <= maxRetries; retry++ {
			if retry > 0 {
				backoffTime := time.Duration(retry*retry) * 500 * time.Millisecond
				log.Printf("Retrying chunk %s (attempt %d/%d) after %v", chunk.PageRange, retry+1, maxRetries+1, backoffTime)
				time.Sleep(backoffTime)
			}

			chunkPrompt := fmt.Sprintf(CHUNK_PROMPT, chunk.Text)
			
			// Create context with timeout for this specific call
			chunkCtx, cancel := context.WithTimeout(ctx, 45*time.Second)
			chunkResp, err := g.flashModel.GenerateContent(chunkCtx, genai.Text(chunkPrompt))
			cancel()

			if err != nil {
				log.Printf("Chunk %s attempt %d failed: %v", chunk.PageRange, retry+1, err)
				if retry == maxRetries {
					consecutiveNoNew++
					break
				}
				continue
			}

			if len(chunkResp.Candidates) == 0 || len(chunkResp.Candidates[0].Content.Parts) == 0 {
				log.Printf("Chunk %s attempt %d returned empty response", chunk.PageRange, retry+1)
				if retry == maxRetries {
					consecutiveNoNew++
					break
				}
				continue
			}

			var chunkResult string
			for _, part := range chunkResp.Candidates[0].Content.Parts {
				if textPart, ok := part.(genai.Text); ok {
					chunkResult += string(textPart)
				}
			}

			log.Printf("RAW preview: %s", truncateStringForSections(strings.ReplaceAll(chunkResult, "\n", " "), 800))
			chunkResult = cleanJSONResponse(chunkResult)

			if json.Unmarshal([]byte(chunkResult), &chunkSections) == nil && len(chunkSections) > 0 {
				chunkResults = append(chunkResults, chunkSections)
				consecutiveNoNew = 0
				success = true
				log.Printf("Chunk %s processed successfully with %d sections", chunk.PageRange, len(chunkSections))
				break
			} else {
				log.Printf("Chunk %s attempt %d failed to parse JSON", chunk.PageRange, retry+1)
				if retry == maxRetries {
					consecutiveNoNew++
				}
			}
		}

		if !success {
			log.Printf("Chunk %s failed after all retry attempts", chunk.PageRange)
		}

		// Early stopping condition
		if consecutiveNoNew >= maxConsecutiveNoNew {
			log.Printf("Early stopping: %d consecutive chunks added no new info.", consecutiveNoNew)
			break
		}

		// Throttle to avoid rate limiting
		if i < len(candidateChunks)-1 {
			time.Sleep(200 * time.Millisecond)
		}
	}

	// Aggregate chunk results
	log.Printf("Processed %d chunks successfully, got %d chunk results", processedCount, len(chunkResults))
	if len(chunkResults) == 0 {
		return &SectionwiseResult{
			Mode:            "chunk_failed",
			Final:           []SectionAnalysis{},
			ProcessedChunks: processedCount,
		}, nil
	}

	// Try model-based aggregation first
	aggregated := g.aggregateChunksWithModel(ctx, chunkResults)
	if aggregated != nil {
		return &SectionwiseResult{
			Mode:  "chunk_optimized",
			Final: *aggregated,
		}, nil
	}

	// Fallback to programmatic aggregation
	final := g.programmaticAggregate(chunkResults)
	return &SectionwiseResult{
		Mode:  "chunk_optimized",
		Final: final,
	}, nil
}

func (g *GeminiService) createDocumentChunks(text string, chunkSize int) []string {
	words := strings.Fields(text)
	chunks := []string{}

	for i := 0; i < len(words); i += chunkSize {
		end := i + chunkSize
		if end > len(words) {
			end = len(words)
		}
		chunk := strings.Join(words[i:end], " ")
		chunks = append(chunks, chunk)
	}

	return chunks
}

func (g *GeminiService) aggregateChunksWithModel(ctx context.Context, chunkResults [][]SectionAnalysis) *[]SectionAnalysis {
	// Convert chunk results to JSON strings
	var jsonArrays []string
	for _, chunk := range chunkResults {
		if jsonBytes, err := json.Marshal(chunk); err == nil {
			jsonArrays = append(jsonArrays, string(jsonBytes))
		}
	}

	if len(jsonArrays) == 0 {
		return nil
	}

	chunksJSON := strings.Join(jsonArrays, "\n")
	prompt := fmt.Sprintf(AGGREGATE_PROMPT, chunksJSON)

	resp, err := g.proModel.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		log.Printf("Model aggregation failed: %v", err)
		return nil
	}

	if len(resp.Candidates) == 0 || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil
	}

	var result string
	for _, part := range resp.Candidates[0].Content.Parts {
		if textPart, ok := part.(genai.Text); ok {
			result += string(textPart)
		}
	}

	result = cleanJSONResponse(result)

	var aggregated []SectionAnalysis
	if json.Unmarshal([]byte(result), &aggregated) == nil {
		return &aggregated
	}

	return nil
}

func (g *GeminiService) programmaticAggregate(chunkResults [][]SectionAnalysis) []SectionAnalysis {
	merged := make(map[string]*SectionAnalysis)

	normalize := func(s string) string {
		re := regexp.MustCompile(`\s+`)
		return strings.ToLower(re.ReplaceAllString(strings.TrimSpace(s), " "))
	}

	for _, chunk := range chunkResults {
		for _, section := range chunk {
			key := normalize(section.SectionName)
			if key == "" {
				key = normalize(section.SectionSummary[:minInt(50, len(section.SectionSummary))])
			}
			if key == "" {
				continue
			}

			if existing, exists := merged[key]; exists {
				// Merge with existing section
				if len(section.SectionSummary) > len(existing.SectionSummary) {
					existing.SectionSummary = section.SectionSummary
				}

				// Deduplicate key considerations
				existingTexts := make(map[string]bool)
				for _, kc := range existing.KeyConsiderations {
					existingTexts[normalize(kc.Consideration)] = true
				}

				for _, kc := range section.KeyConsiderations {
					if !existingTexts[normalize(kc.Consideration)] {
						existing.KeyConsiderations = append(existing.KeyConsiderations, kc)
						existingTexts[normalize(kc.Consideration)] = true
					}
				}
			} else {
				// Create new section
				merged[key] = &SectionAnalysis{
					SectionName:       section.SectionName,
					SectionSummary:    section.SectionSummary,
					KeyConsiderations: section.KeyConsiderations,
				}
			}
		}
	}

	// Convert map to slice
	result := make([]SectionAnalysis, 0, len(merged))
	for _, section := range merged {
		result = append(result, *section)
	}

	return result
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func truncateStringForSections(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// Section header keywords for filtering
var sectionHeaderKeywords = []string{
	`\\brfp\\b`, `\\bsection\\b`, `\\bscope\\b`, `\\bscope of work\\b`, `\\bproject overview\\b`,
	`\\bmajor work\\b`, `\\btechnical standard\\b`, `\\bsection-wise\\b`, `\\beligibility\\b`,
	`\\bsection wise\\b`, `\\brfp section\\b`,
}

func (g *GeminiService) extractTextByPage(documentText string) []string {
	// Split document by page markers [PAGE:X]
	pageRegex := regexp.MustCompile(`\\[PAGE:(\\d+)\\]`)
	parts := pageRegex.Split(documentText, -1)
	pages := make([]string, 0)

	for i, part := range parts {
		if i == 0 && !pageRegex.MatchString(documentText[:len(part)]) {
			// First part before any page marker
			continue
		}
		pages = append(pages, strings.TrimSpace(part))
	}

	// Fallback: if no page markers found, split by estimated page size
	if len(pages) == 0 {
		estimatedPageSize := 3000 // chars per page
		for i := 0; i < len(documentText); i += estimatedPageSize {
			end := i + estimatedPageSize
			if end > len(documentText) {
				end = len(documentText)
			}
			pages = append(pages, documentText[i:end])
		}
	}

	return pages
}

func (g *GeminiService) makeChunksFromPages(pageTexts []string, pagesPerChunk, overlapPages int) []OptimizedChunk {
	chunks := make([]OptimizedChunk, 0)
	n := len(pageTexts)
	if n == 0 {
		return chunks
	}

	i := 0
	for i < n {
		start := i
		end := minInt(n, i+pagesPerChunk)

		// Build chunk text with page markers
		var textParts []string
		for idx := start; idx < end; idx++ {
			textParts = append(textParts, fmt.Sprintf("[PAGE:%d]\\n%s", idx+1, pageTexts[idx]))
		}
		text := strings.Join(textParts, "\\n\\n")

		chunks = append(chunks, OptimizedChunk{
			StartPage: start + 1,
			EndPage:   end,
			Text:      text,
			PageRange: fmt.Sprintf("%d-%d", start+1, end),
		})

		if end == n {
			break
		}
		i = end - overlapPages
	}

	return chunks
}

func (g *GeminiService) filterCandidateChunks(chunks []OptimizedChunk) []OptimizedChunk {
	candidates := make([]OptimizedChunk, 0)

	for _, chunk := range chunks {
		if g.chunkLikelyHasSectionHeader(chunk.Text) {
			candidates = append(candidates, chunk)
		}
	}

	// If no candidates found, use all chunks
	if len(candidates) == 0 {
		return chunks
	}

	return candidates
}

func (g *GeminiService) chunkLikelyHasSectionHeader(chunkText string) bool {
	s := strings.ToLower(chunkText)

	// Check for section header keywords
	for _, pattern := range sectionHeaderKeywords {
		if matched, _ := regexp.MatchString(pattern, s); matched {
			return true
		}
	}

	// Heuristic: detect all-caps headings
	lines := strings.Split(chunkText, "\\n")
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if len(line) >= 4 && len(line) <= 120 && line == strings.ToUpper(line) && len(strings.Fields(line)) < 12 {
			return true
		}
	}

	return false
}
