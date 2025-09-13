package main

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"sort"
	"strings"
	"sync"
)

// Simple in-memory vector store for document embeddings
type VectorStore struct {
	documents map[string]*Document
	mutex     sync.RWMutex
}

type Document struct {
	ID       string                 `json:"id"`
	Content  string                 `json:"content"`
	Metadata map[string]interface{} `json:"metadata"`
	Chunks   []DocumentChunk        `json:"chunks"`
}

type DocumentChunk struct {
	ID        string    `json:"id"`
	Content   string    `json:"content"`
	Embedding []float64 `json:"embedding"`
	PageNum   int       `json:"page_num"`
}

type SearchResult struct {
	ChunkID    string                 `json:"chunk_id"`
	Content    string                 `json:"content"`
	Score      float64                `json:"score"`
	DocumentID string                 `json:"document_id"`
	Metadata   map[string]interface{} `json:"metadata"`
}

func NewVectorStore() *VectorStore {
	return &VectorStore{
		documents: make(map[string]*Document),
	}
}

func (vs *VectorStore) AddDocument(content string, metadata map[string]interface{}) (string, error) {
	vs.mutex.Lock()
	defer vs.mutex.Unlock()

	// Generate document ID
	docID := fmt.Sprintf("%x", md5.Sum([]byte(content)))

	// Parse PDF and create chunks
	parser := NewPDFParser()
	chunks := parser.ChunkText(content, 1000) // 1000 character chunks

	var documentChunks []DocumentChunk
	for i, chunk := range chunks {
		chunkID := fmt.Sprintf("%s_chunk_%d", docID, i)
		
		// Generate simple embedding (in production, use a real embedding model)
		embedding := vs.generateSimpleEmbedding(chunk)
		
		documentChunks = append(documentChunks, DocumentChunk{
			ID:        chunkID,
			Content:   chunk,
			Embedding: embedding,
			PageNum:   i + 1, // Approximate page number
		})
	}

	document := &Document{
		ID:       docID,
		Content:  content,
		Metadata: metadata,
		Chunks:   documentChunks,
	}

	vs.documents[docID] = document
	log.Printf("Added document %s with %d chunks", docID, len(documentChunks))

	return docID, nil
}

func (vs *VectorStore) SearchSimilar(query string, topK int) ([]SearchResult, error) {
	vs.mutex.RLock()
	defer vs.mutex.RUnlock()

	if topK <= 0 {
		topK = 5
	}

	queryEmbedding := vs.generateSimpleEmbedding(query)
	var results []SearchResult

	for docID, doc := range vs.documents {
		for _, chunk := range doc.Chunks {
			similarity := vs.cosineSimilarity(queryEmbedding, chunk.Embedding)
			
			results = append(results, SearchResult{
				ChunkID:    chunk.ID,
				Content:    chunk.Content,
				Score:      similarity,
				DocumentID: docID,
				Metadata:   doc.Metadata,
			})
		}
	}

	// Sort by similarity score (descending)
	sort.Slice(results, func(i, j int) bool {
		return results[i].Score > results[j].Score
	})

	// Return top K results
	if len(results) > topK {
		results = results[:topK]
	}

	return results, nil
}

func (vs *VectorStore) GetDocument(docID string) (*Document, bool) {
	vs.mutex.RLock()
	defer vs.mutex.RUnlock()

	doc, exists := vs.documents[docID]
	return doc, exists
}

func (vs *VectorStore) DeleteDocument(docID string) bool {
	vs.mutex.Lock()
	defer vs.mutex.Unlock()

	_, exists := vs.documents[docID]
	if exists {
		delete(vs.documents, docID)
	}
	return exists
}

func (vs *VectorStore) ListDocuments() []string {
	vs.mutex.RLock()
	defer vs.mutex.RUnlock()

	var docIDs []string
	for docID := range vs.documents {
		docIDs = append(docIDs, docID)
	}
	return docIDs
}

// Simple embedding generation (in production, use a real embedding model like OpenAI's)
func (vs *VectorStore) generateSimpleEmbedding(text string) []float64 {
	// This is a very simple embedding based on word frequency
	// In production, you should use proper embedding models
	words := strings.Fields(strings.ToLower(text))
	wordCount := make(map[string]int)
	
	for _, word := range words {
		// Simple preprocessing
		word = strings.Trim(word, ".,!?;:")
		if len(word) > 2 {
			wordCount[word]++
		}
	}

	// Create a fixed-size embedding vector (100 dimensions)
	embedding := make([]float64, 100)
	
	// Simple hash-based embedding
	for word, count := range wordCount {
		hash := 0
		for _, char := range word {
			hash = (hash*31 + int(char)) % 100
		}
		if hash < 0 {
			hash = -hash
		}
		embedding[hash] += float64(count)
	}

	// Normalize the embedding
	norm := 0.0
	for _, val := range embedding {
		norm += val * val
	}
	norm = math.Sqrt(norm)
	
	if norm > 0 {
		for i := range embedding {
			embedding[i] /= norm
		}
	}

	return embedding
}

// Calculate cosine similarity between two vectors
func (vs *VectorStore) cosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) {
		return 0.0
	}

	var dotProduct, normA, normB float64
	
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// Export document data as JSON
func (vs *VectorStore) ExportDocument(docID string) ([]byte, error) {
	vs.mutex.RLock()
	defer vs.mutex.RUnlock()

	doc, exists := vs.documents[docID]
	if !exists {
		return nil, fmt.Errorf("document not found")
	}

	return json.MarshalIndent(doc, "", "  ")
}
