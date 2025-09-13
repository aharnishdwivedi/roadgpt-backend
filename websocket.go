package main

import (
	"log"
	"net/http"

	"github.com/gorilla/websocket"
	"github.com/labstack/echo/v4"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool {
		// Allow connections from any origin for development
		// In production, you should restrict this to your frontend domain
		return true
	},
}

type WebSocketHandler struct {
	openAIService *OpenAIService
}

type Message struct {
	Type    string `json:"type"`
	Content string `json:"content"`
	Error   string `json:"error,omitempty"`
}

func NewWebSocketHandler(openAIService *OpenAIService) *WebSocketHandler {
	return &WebSocketHandler{
		openAIService: openAIService,
	}
}

func (h *WebSocketHandler) HandleWebSocket(c echo.Context) error {
	ws, err := upgrader.Upgrade(c.Response(), c.Request(), nil)
	if err != nil {
		log.Printf("WebSocket upgrade error: %v", err)
		return err
	}
	defer ws.Close()

	log.Println("New WebSocket connection established")

	// Send welcome message
	welcomeMsg := Message{
		Type:    "system",
		Content: "Connected to RoadGPT! Ask me anything about road safety, traffic, or driving.",
	}
	if err := ws.WriteJSON(welcomeMsg); err != nil {
		log.Printf("Error sending welcome message: %v", err)
		return err
	}

	for {
		var msg Message
		err := ws.ReadJSON(&msg)
		if err != nil {
			if websocket.IsUnexpectedCloseError(err, websocket.CloseGoingAway, websocket.CloseAbnormalClosure) {
				log.Printf("WebSocket error: %v", err)
			}
			break
		}

		log.Printf("Received message: %s", msg.Content)

		// Process the message based on type
		switch msg.Type {
		case "user_message":
			go h.handleUserMessage(ws, msg.Content)
		case "ping":
			pongMsg := Message{Type: "pong", Content: "pong"}
			if err := ws.WriteJSON(pongMsg); err != nil {
				log.Printf("Error sending pong: %v", err)
				return err
			}
		default:
			errorMsg := Message{
				Type:  "error",
				Error: "Unknown message type",
			}
			if err := ws.WriteJSON(errorMsg); err != nil {
				log.Printf("Error sending error message: %v", err)
				return err
			}
		}
	}

	return nil
}

func (h *WebSocketHandler) handleUserMessage(ws *websocket.Conn, userMessage string) {
	// Send typing indicator
	typingMsg := Message{
		Type:    "typing",
		Content: "RoadGPT is thinking...",
	}
	if err := ws.WriteJSON(typingMsg); err != nil {
		log.Printf("Error sending typing indicator: %v", err)
		return
	}

	// Get response from OpenAI
	response, err := h.openAIService.GetChatResponse(userMessage)
	if err != nil {
		errorMsg := Message{
			Type:  "error",
			Error: "Sorry, I'm having trouble processing your request. Please try again.",
		}
		if err := ws.WriteJSON(errorMsg); err != nil {
			log.Printf("Error sending error response: %v", err)
		}
		return
	}

	// Send the AI response
	responseMsg := Message{
		Type:    "ai_response",
		Content: response,
	}
	if err := ws.WriteJSON(responseMsg); err != nil {
		log.Printf("Error sending AI response: %v", err)
	}
}
