package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type GeminiService struct {
	client    *genai.Client
	proModel  *genai.GenerativeModel
	flashModel *genai.GenerativeModel
}

func NewGeminiService(apiKey string) *GeminiService {
	if apiKey == "" {
		log.Println("Warning: Gemini API key not provided. Set GEMINI_API_KEY environment variable.")
		return &GeminiService{}
	}

	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		log.Printf("Failed to create Gemini client: %v", err)
		return &GeminiService{}
	}

	proModel := client.GenerativeModel("gemini-2.5-pro")
	proModel.SetTemperature(0.7)
	proModel.SetMaxOutputTokens(8192)

	flashModel := client.GenerativeModel("gemini-2.5-flash")
	flashModel.SetTemperature(0.7)
	flashModel.SetMaxOutputTokens(8192)

	return &GeminiService{
		client:     client,
		proModel:   proModel,
		flashModel: flashModel,
	}
}

func (g *GeminiService) AnalyzeTenderDocument(documentText string, query string) (string, error) {
	if g.client == nil || g.proModel == nil || g.flashModel == nil {
		return "Gemini service is not properly configured. Please check the API key.", fmt.Errorf("Gemini client not initialized")
	}

	ctx := context.Background()

	// Create a specialized prompt for tender document analysis
	prompt := fmt.Sprintf("You are an expert tender document analyst with deep knowledge of government procurement processes. Analyze the following tender document comprehensively and extract ALL available information in the exact JSON format specified below.\n\nIMPORTANT INSTRUCTIONS:\n1. Extract ONLY information explicitly mentioned in the document\n2. For dates, look for patterns like dd/mm/yyyy, dd-mm-yyyy, or written dates\n3. For financial amounts, look for currency symbols, Rs, ₹, Crore, Lakh, etc.\n4. For percentages, look for %% symbol or written percentages\n5. If information is not found, use 'Not specified in provided text'\n6. Be thorough - scan the entire document for scattered information\n\nDocument content: %s\n\nUser query: %s\n\nPlease respond with ONLY a valid JSON object in this exact format:\n{\n  \"tender_id\": \"exact tender/RFP/NIT number from document header or title\",\n  \"title\": \"complete project title as mentioned in the document\",\n  \"due_date\": \"bid submission deadline with exact date and time\",\n  \"issuing_authority\": \"full name of issuing organization/department\",\n  \"contract_value\": \"total estimated project cost with currency\",\n  \"project_overview\": \"comprehensive description of project scope, deliverables, and objectives from the document\",\n  \"financial_requirements\": {\n    \"contract_value\": \"total contract value with currency if different from above\",\n    \"emd\": \"earnest money deposit amount and percentage of contract value\",\n    \"performance_bg\": \"performance bank guarantee amount and percentage\",\n    \"document_fees\": \"tender document purchase cost if mentioned\"\n  },\n  \"eligibility_highlights\": [\n    \"minimum experience requirements in years\",\n    \"annual turnover requirements with amounts\",\n    \"technical qualifications needed\",\n    \"registration/license requirements\",\n    \"equipment requirements if any\"\n  ],\n  \"important_dates\": {\n    \"pre_bid_queries\": \"last date for pre-bid queries with date and time\",\n    \"bid_submission\": \"bid submission deadline with date and time\",\n    \"technical_bid_opening\": \"technical bid opening date and time\",\n    \"financial_bid_opening\": \"financial bid opening date and time if mentioned\"\n  }\n}", documentText, query)

	// Try Gemini 2.5 Pro first
	log.Printf("Attempting analysis with Gemini 2.5 Pro...")
	resp, err := g.proModel.GenerateContent(ctx, genai.Text(prompt))
	
	if err != nil {
		log.Printf("Gemini 2.5 Pro failed: %v, falling back to Flash", err)
		// Fallback to Flash
		log.Printf("Falling back to Gemini 2.5 Flash...")
		resp, err = g.flashModel.GenerateContent(ctx, genai.Text(prompt))
		if err != nil {
			log.Printf("Both Gemini models failed: %v", err)
			return "I'm having trouble analyzing the document right now. This could be due to an API issue or temporary service unavailability. Please try again.", fmt.Errorf("failed to get response from both Gemini models: %w", err)
		}
	} else {
		log.Printf("Gemini 2.5 Pro succeeded")
	}
	
	log.Printf("Received response from Gemini. Candidates count: %d", len(resp.Candidates))

	if len(resp.Candidates) == 0 {
		log.Printf("Gemini response has no candidates. Full response: %+v", resp)
		return "I received an empty response from the analysis service. Please try again.", fmt.Errorf("no candidates returned from Gemini")
	}
	
	if len(resp.Candidates[0].Content.Parts) == 0 {
		log.Printf("Gemini candidate has no content parts. Candidate: %+v", resp.Candidates[0])
		if resp.Candidates[0].FinishReason.String() == "FinishReasonMaxTokens" {
			log.Printf("Response was truncated due to max tokens limit")
			return "The document analysis was truncated due to length. Please try with a shorter document or contact support.", fmt.Errorf("response truncated due to max tokens")
		}
		return "I received an empty response from the analysis service. Please try again.", fmt.Errorf("no content parts returned from Gemini")
	}

	// Extract text from the response
	var result string
	for _, part := range resp.Candidates[0].Content.Parts {
		if textPart, ok := part.(genai.Text); ok {
			result += string(textPart)
		}
	}

	// Clean the response - remove markdown code blocks if present
	result = cleanJSONResponse(result)

	return result, nil
}

// cleanJSONResponse removes markdown code blocks and cleans up the JSON response
func cleanJSONResponse(response string) string {
	// Remove markdown code blocks (```json and ```)
	response = strings.ReplaceAll(response, "```json", "")
	response = strings.ReplaceAll(response, "```", "")
	
	// Trim whitespace
	response = strings.TrimSpace(response)
	
	return response
}

func (g *GeminiService) Close() {
	if g.client != nil {
		g.client.Close()
	}
}
