package main

import (
	"context"
	"fmt"
	"log"

	"github.com/sashabaranov/go-openai"
)

type OpenAIService struct {
	client *openai.Client
}

func NewOpenAIService(apiKey string) *OpenAIService {
	if apiKey == "" {
		log.Println("Warning: OpenAI API key not provided. Set OPENAI_API_KEY environment variable.")
	}
	
	client := openai.NewClient(apiKey)
	return &OpenAIService{
		client: client,
	}
}

func (s *OpenAIService) GetChatResponse(userMessage string) (string, error) {
	if s.client == nil {
		return "", fmt.Errorf("OpenAI client not initialized")
	}

	// Create a system prompt focused on road safety and driving
	systemPrompt := `You are RoadGPT, an AI assistant specialized in road safety, traffic management, driving tips, and transportation-related topics. 

Your expertise includes:
- Road safety guidelines and best practices
- Traffic rules and regulations
- Defensive driving techniques
- Vehicle maintenance tips
- Weather-related driving advice
- Emergency procedures on the road
- Transportation infrastructure
- Traffic flow optimization
- Accident prevention strategies

Always provide helpful, accurate, and safety-focused responses. If asked about topics outside your expertise, politely redirect the conversation back to road and transportation topics.`

	resp, err := s.client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT3Dot5Turbo,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleSystem,
					Content: systemPrompt,
				},
				{
					Role:    openai.ChatMessageRoleUser,
					Content: userMessage,
				},
			},
			MaxTokens:   500,
			Temperature: 0.7,
		},
	)

	if err != nil {
		log.Printf("OpenAI API error: %v", err)
		return "", fmt.Errorf("failed to get response from OpenAI: %w", err)
	}

	if len(resp.Choices) == 0 {
		return "", fmt.Errorf("no response choices returned from OpenAI")
	}

	return resp.Choices[0].Message.Content, nil
}
