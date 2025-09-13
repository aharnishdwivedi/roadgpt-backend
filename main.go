package main

import (
	"log"
	"net/http"
	"os"

	"github.com/labstack/echo/v4"
	"github.com/labstack/echo/v4/middleware"
)

func main() {
	// Create Echo instance
	e := echo.New()

	// Middleware
	e.Use(middleware.Logger())
	e.Use(middleware.Recover())
	e.Use(middleware.CORS())

	// Initialize services
	openAIService := NewOpenAIService(os.Getenv("OPENAI_API_KEY"))
	wsHandler := NewWebSocketHandler(openAIService)

	// Routes
	e.GET("/", func(c echo.Context) error {
		return c.String(http.StatusOK, "RoadGPT Backend Server is running!")
	})

	// WebSocket endpoint for roadgpt
	e.GET("/roadgpt", wsHandler.HandleWebSocket)

	// Health check endpoint
	e.GET("/health", func(c echo.Context) error {
		return c.JSON(http.StatusOK, map[string]string{
			"status": "healthy",
			"service": "roadgpt-backend",
		})
	})

	// Start server
	port := os.Getenv("PORT")
	if port == "" {
		port = "8082"
	}

	log.Printf("Server starting on port %s", port)
	log.Fatal(e.Start(":" + port))
}
