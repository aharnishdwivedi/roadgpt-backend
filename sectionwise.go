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
	SectionName        string              `json:"section_name"`
	SectionSummary     string              `json:"section_summary"`
	KeyConsiderations  []KeyConsideration  `json:"key_considerations"`
}

type KeyConsideration struct {
	Text     string `json:"text"`
	Critical bool   `json:"critical"`
	Page     string `json:"page"`
}

type SectionwiseResult struct {
	Mode     string            `json:"mode"`
	Final    []SectionAnalysis `json:"final"`
	RawSingle string           `json:"raw_single,omitempty"`
}

const SINGLE_DOC_PROMPT = `You are an expert document parser. From the DOCUMENT extract every logical section and return a JSON array of section objects.

Each section object must have exactly these keys:
• "section_name": short title of the section (string)
• "section_summary": concise 1-3 sentence summary of the section content (do NOT invent; summarize only what's present)
• "key_considerations": array of objects, each with:
    { "text": "...", "critical": true|false, "page": "page number or page range where this appears" }

Rules:
• RESPOND WITH A SINGLE VALID JSON ARRAY (list) and nothing else.
• Mark "critical": true if the language explicitly flags it as critical/mandatory/penalty/limit OR if it contains strong actionable items (deadlines, penalties, percentages, "CRITICAL", "must", "shall", "mandatory").
• Include page numbers wherever you can (use page markers in the DOCUMENT).
• If you cannot find sections, return an empty list [].

DOCUMENT:
%s`

const CHUNK_PROMPT = `You are an expert document parser. From the DOCUMENT CHUNK extract all sections found in this chunk and return a single VALID JSON ARRAY of section objects with keys:
• section_name
• section_summary  
• key_considerations: [ { "text": "...", "critical": true|false, "page": "..." }, ... ]

Rules:
• Return JSON ONLY (no explanation).
• Use page markers in the chunk for provenance.

DOCUMENT CHUNK:
%s`

const AGGREGATE_PROMPT = `You are given multiple JSON arrays (chunk-level extractions) representing sections found across a document. Combine them into one consolidated JSON array of sections.

Rules:
• For sections with the same or very similar names, merge them into one section; keep the longest/best summary, and combine key_considerations, deduplicating exact duplicate items (case-insensitive).
• Keep page provenance for every key_consideration.
• Do NOT invent facts.
• Return a single VALID JSON ARRAY of section objects with keys: section_name, section_summary, key_considerations.

Chunk-level JSON arrays (one per chunk):
%s`

func (g *GeminiService) ExtractSectionwiseAnalysis(documentText string) (*SectionwiseResult, error) {
	if g.client == nil || g.proModel == nil || g.flashModel == nil {
		return nil, fmt.Errorf("Gemini client not initialized")
	}

	ctx := context.Background()

	// 1. Try single-call extraction with Pro first, fallback to Flash
	log.Printf("Attempting single-call section-wise extraction with Gemini 2.5 Pro...")
	
	prompt := fmt.Sprintf(SINGLE_DOC_PROMPT, documentText)
	resp, err := g.proModel.GenerateContent(ctx, genai.Text(prompt))
	
	if err != nil {
		log.Printf("Gemini 2.5 Pro failed for sections: %v, falling back to Flash", err)
		resp, err = g.flashModel.GenerateContent(ctx, genai.Text(prompt))
		if err != nil {
			log.Printf("Both models failed for sections: %v", err)
		}
	}
	
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
			log.Printf("Single-call extraction successful, found %d sections", len(sections))
			return &SectionwiseResult{
				Mode:      "single_call",
				Final:     sections,
				RawSingle: result,
			}, nil
		}
		
		log.Printf("Single-call returned invalid JSON, falling back to chunked approach")
	} else {
		log.Printf("Single-call failed: %v", err)
	}

	// 2. Fallback to chunked extraction
	log.Printf("Starting chunked section-wise extraction...")
	
	chunks := g.createDocumentChunks(documentText, 3000) // ~3000 chars per chunk
	chunkResults := [][]SectionAnalysis{}
	
	for i, chunk := range chunks {
		log.Printf("Processing chunk %d/%d", i+1, len(chunks))
		
		chunkPrompt := fmt.Sprintf(CHUNK_PROMPT, chunk)
		chunkResp, err := g.flashModel.GenerateContent(ctx, genai.Text(chunkPrompt))
		
		if err != nil {
			log.Printf("Chunk %d failed: %v", i+1, err)
			continue
		}
		
		if len(chunkResp.Candidates) == 0 || len(chunkResp.Candidates[0].Content.Parts) == 0 {
			log.Printf("Chunk %d returned empty response", i+1)
			continue
		}
		
		var chunkResult string
		for _, part := range chunkResp.Candidates[0].Content.Parts {
			if textPart, ok := part.(genai.Text); ok {
				chunkResult += string(textPart)
			}
		}
		
		chunkResult = cleanJSONResponse(chunkResult)
		
		var chunkSections []SectionAnalysis
		if json.Unmarshal([]byte(chunkResult), &chunkSections) == nil {
			chunkResults = append(chunkResults, chunkSections)
		}
		
		// Small delay to avoid rate limiting
		time.Sleep(500 * time.Millisecond)
	}
	
	// 3. Aggregate chunk results
	if len(chunkResults) == 0 {
		return &SectionwiseResult{
			Mode:  "chunk_failed",
			Final: []SectionAnalysis{},
		}, nil
	}
	
	// Try model-based aggregation first
	aggregated := g.aggregateChunksWithModel(ctx, chunkResults)
	if aggregated != nil {
		return &SectionwiseResult{
			Mode:  "chunk_model_aggregate",
			Final: *aggregated,
		}, nil
	}
	
	// Fallback to programmatic aggregation
	final := g.programmaticAggregate(chunkResults)
	return &SectionwiseResult{
		Mode:  "chunk_programmatic_aggregate", 
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
					existingTexts[normalize(kc.Text)] = true
				}
				
				for _, kc := range section.KeyConsiderations {
					if !existingTexts[normalize(kc.Text)] {
						existing.KeyConsiderations = append(existing.KeyConsiderations, kc)
						existingTexts[normalize(kc.Text)] = true
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

