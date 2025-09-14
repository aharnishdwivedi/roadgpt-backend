package main

import (
	"fmt"
	"io"
	"strings"

	"github.com/gen2brain/go-fitz"
)

type PDFParser struct{}

func NewPDFParser() *PDFParser {
	return &PDFParser{}
}

func (p *PDFParser) ExtractText(reader io.ReaderAt, size int64) (string, error) {
	// Read all data from ReaderAt
	data := make([]byte, size)
	_, err := reader.ReadAt(data, 0)
	if err != nil {
		return "", fmt.Errorf("failed to read PDF data: %w", err)
	}

	// Open PDF document from memory
	doc, err := fitz.NewFromMemory(data)
	if err != nil {
		return "", fmt.Errorf("failed to open PDF document: %w", err)
	}
	defer doc.Close()

	var textBuilder strings.Builder
	
	// Extract text from each page
	for pageNum := 0; pageNum < doc.NumPage(); pageNum++ {
		text, err := doc.Text(pageNum)
		if err != nil {
			continue // Skip pages with errors
		}
		
		if strings.TrimSpace(text) != "" {
			textBuilder.WriteString(fmt.Sprintf("\n--- Page %d ---\n", pageNum+1))
			textBuilder.WriteString(text)
			textBuilder.WriteString("\n")
		}
	}

	extractedText := textBuilder.String()
	if len(strings.TrimSpace(extractedText)) == 0 {
		return "", fmt.Errorf("no text content found in PDF")
	}

	return extractedText, nil
}

// ChunkText splits text into smaller chunks for better processing
func (p *PDFParser) ChunkText(text string, maxChunkSize int) []string {
	if maxChunkSize <= 0 {
		maxChunkSize = 4000 // Default chunk size
	}

	words := strings.Fields(text)
	var chunks []string
	var currentChunk strings.Builder

	for _, word := range words {
		// Check if adding this word would exceed the chunk size
		if currentChunk.Len()+len(word)+1 > maxChunkSize {
			if currentChunk.Len() > 0 {
				chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
				currentChunk.Reset()
			}
		}
		
		if currentChunk.Len() > 0 {
			currentChunk.WriteString(" ")
		}
		currentChunk.WriteString(word)
	}

	// Add the last chunk if it has content
	if currentChunk.Len() > 0 {
		chunks = append(chunks, strings.TrimSpace(currentChunk.String()))
	}

	return chunks
}

// ExtractMetadata extracts basic metadata from PDF
func (p *PDFParser) ExtractMetadata(reader io.ReaderAt, size int64) (map[string]interface{}, error) {
	// Read all data from ReaderAt
	data := make([]byte, size)
	_, err := reader.ReadAt(data, 0)
	if err != nil {
		return nil, fmt.Errorf("failed to read PDF data: %w", err)
	}

	// Open PDF document from memory
	doc, err := fitz.NewFromMemory(data)
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF document: %w", err)
	}
	defer doc.Close()

	metadata := make(map[string]interface{})
	metadata["num_pages"] = doc.NumPage()
	
	// Extract document metadata
	if meta := doc.Metadata(); meta != nil {
		if title := meta["title"]; title != "" {
			metadata["title"] = title
		}
		if author := meta["author"]; author != "" {
			metadata["author"] = author
		}
		if subject := meta["subject"]; subject != "" {
			metadata["subject"] = subject
		}
		metadata["has_info"] = true
	}

	return metadata, nil
}

// ExtractTextByPage extracts text from each page separately
func (p *PDFParser) ExtractTextByPage(reader io.Reader) ([]string, error) {
	// Read all data from reader
	data, err := io.ReadAll(reader)
	if err != nil {
		return nil, fmt.Errorf("failed to read PDF data: %w", err)
	}

	// Open PDF document from memory
	doc, err := fitz.NewFromMemory(data)
	if err != nil {
		return nil, fmt.Errorf("failed to open PDF document: %w", err)
	}
	defer doc.Close()

	var pages []string
	
	// Extract text from each page
	for pageNum := 0; pageNum < doc.NumPage(); pageNum++ {
		text, err := doc.Text(pageNum)
		if err != nil {
			pages = append(pages, "") // Add empty page on error
			continue
		}
		pages = append(pages, text)
	}

	return pages, nil
}
