# RoadGPT Backend

A Go Echo backend server with WebSocket support for real-time AI chat functionality, specifically designed for road safety and transportation-related queries.

## Features

- **WebSocket Communication**: Real-time bidirectional communication between frontend and backend
- **OpenAI Integration**: Powered by OpenAI's GPT-3.5-turbo for intelligent responses
- **Road Safety Focus**: Specialized AI assistant for road safety, traffic, and driving topics
- **CORS Enabled**: Ready for cross-origin requests from frontend applications
- **Health Check**: Built-in health monitoring endpoint

## Prerequisites

- Go 1.21 or higher
- OpenAI API key

## Installation

1. Clone or navigate to the project directory:
```bash
cd roadgpt-backend
```

2. Install dependencies:
```bash
go mod tidy
```

3. Set up environment variables:
```bash
cp .env.example .env
```

4. Edit `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_actual_openai_api_key_here
PORT=8080
```

## Running the Server

### Development
```bash
# Set environment variable (or use .env file)
export OPENAI_API_KEY=your_openai_api_key_here

# Run the server
go run .
```

### Production Build
```bash
# Build the binary
go build -o roadgpt-backend

# Run the binary
./roadgpt-backend
```

## API Endpoints

### WebSocket Endpoint
- **URL**: `ws://localhost:8080/roadgpt`
- **Protocol**: WebSocket
- **Purpose**: Real-time chat with RoadGPT AI

### HTTP Endpoints
- **GET /**: Welcome message
- **GET /health**: Health check endpoint

## WebSocket Message Format

### Client to Server Messages
```json
{
  "type": "user_message",
  "content": "What are the best practices for driving in rain?"
}

{
  "type": "ping",
  "content": "ping"
}
```

### Server to Client Messages
```json
{
  "type": "system",
  "content": "Connected to RoadGPT! Ask me anything about road safety, traffic, or driving."
}

{
  "type": "ai_response",
  "content": "Here are the best practices for driving in rain..."
}

{
  "type": "typing",
  "content": "RoadGPT is thinking..."
}

{
  "type": "error",
  "error": "Error message here"
}

{
  "type": "pong",
  "content": "pong"
}
```

## Frontend Integration

To connect from your frontend (e.g., React, Vue, vanilla JS):

```javascript
const ws = new WebSocket('ws://localhost:8080/roadgpt');

ws.onopen = function() {
    console.log('Connected to RoadGPT');
};

ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    console.log('Received:', message);
    
    switch(message.type) {
        case 'ai_response':
            // Display AI response in your UI
            break;
        case 'typing':
            // Show typing indicator
            break;
        case 'error':
            // Handle error
            break;
    }
};

// Send a message
function sendMessage(content) {
    ws.send(JSON.stringify({
        type: 'user_message',
        content: content
    }));
}
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key (required) | - |
| `PORT` | Server port | 8080 |

## Project Structure

```
roadgpt-backend/
├── main.go          # Main server setup and routing
├── websocket.go     # WebSocket handler and message processing
├── openai.go        # OpenAI API integration
├── go.mod           # Go module dependencies
├── .env.example     # Environment variables template
└── README.md        # This file
```

## Development Notes

- The server uses CORS middleware to allow cross-origin requests
- WebSocket connections are upgraded from HTTP requests
- The AI is specifically prompted to focus on road safety and transportation topics
- Error handling is implemented for both WebSocket and OpenAI API failures
- Logging is enabled for debugging and monitoring

## Troubleshooting

1. **"OpenAI API key not provided"**: Make sure you've set the `OPENAI_API_KEY` environment variable
2. **WebSocket connection fails**: Check that the server is running on the correct port and CORS is properly configured
3. **AI responses are slow**: This is normal for OpenAI API calls; the typing indicator helps manage user expectations

## License

This project is part of the RoadVision Dashboard system.
