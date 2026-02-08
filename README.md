# Document Q&A AI Agent
An enterprise-ready AI agent for intelligent document question-answering using Large Language Models.

## 🌟 Features

- **Multi-document PDF Processing**: Ingest and process multiple PDF documents with high accuracy
- **Smart Text Extraction**: Extract titles, abstracts, sections, tables, and references
- **Intelligent Q&A Interface**: 
  - Direct content lookup
  - Summarization of key insights
  - Extraction of specific metrics and evaluation results
- **ArXiv Integration**: Lookup and integrate papers from ArXiv (Bonus Feature)
- **Enterprise-Grade Features**: 
  - Context optimization
  - Response caching
  - Rate limiting
  - Secure API design

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key or Gemini API key
- 4GB RAM minimum (8GB recommended)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd document-qa-agent
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Configure environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. Run the application:
```bash
python main.py
```

6. Open browser and navigate to: `http://localhost:8000`

## 📁 Project Structure

```
document-qa-agent/
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables template
├── README.md                  # This file
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration management
│   ├── document_processor.py   # PDF text extraction
│   ├── llm_service.py         # LLM integration (OpenAI/Gemini)
│   ├── qa_engine.py           # Q&A processing engine
│   ├── arxiv_service.py       # ArXiv API integration
│   ├── cache_manager.py       # Response caching
│   └── utils.py               # Utility functions
├── api/
│   ├── __init__.py
│   ├── routes.py              # FastAPI routes
│   └── models.py              # Pydantic models
├── static/
│   ├── index.html             # Web interface
│   └── styles.css             # Styling
├── uploads/                   # Uploaded documents
└── data/                      # Processed data storage
```

## 🔧 Configuration

Edit `.env` file with your settings:

```env
# API Configuration
OPENAI_API_KEY=your_openai_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here

# LLM Settings
LLM_PROVIDER=openai  # or 'gemini'
MODEL_NAME=gpt-4o-mini
TEMPERATURE=0.1
MAX_TOKENS=2000

# Application Settings
HOST=0.0.0.0
PORT=8000
DEBUG=False

# Caching
CACHE_ENABLED=True
CACHE_TTL=3600

# Rate Limiting
RATE_LIMIT_REQUESTS=100
RATE_LIMIT_WINDOW=60
```

## 📡 API Endpoints

### Documents
- `POST /api/documents/upload` - Upload PDF document
- `GET /api/documents` - List all documents
- `DELETE /api/documents/{id}` - Delete document

### Q&A
- `POST /api/qa/query` - Ask a question
- `GET /api/qa/history` - Get query history

### ArXiv (Bonus)
- `POST /api/arxiv/search` - Search ArXiv papers
- `POST /api/arxiv/ingest` - Ingest ArXiv paper

### Health
- `GET /health` - Health check

## 💡 Usage Examples

### Upload a Document
```bash
curl -X POST -F "file=@document.pdf" http://localhost:8000/api/documents/upload
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/api/qa/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main conclusion of the paper?", "document_ids": ["doc1"]}'
```

### Search ArXiv
```bash
curl -X POST http://localhost:8000/api/arxiv/search \
  -H "Content-Type: application/json" \
  -d '{"query": "transformer attention mechanism", "max_results": 5}'
```

## 🎯 Key Features Implementation

### 1. Document Processing Pipeline
- Multi-page PDF text extraction
- Table detection and extraction
- Structure recognition (titles, sections, paragraphs)
- Equation and figure preservation

### 2. LLM Integration
- Support for OpenAI GPT-4o and Gemini models
- Context window optimization
- Token management and truncation
- Streaming responses

### 3. Enterprise Features
- Response caching for common queries
- Rate limiting to prevent abuse
- Request queuing and prioritization
- Error handling and logging

### 4. User Interface
- Clean, responsive web interface
- Drag-and-drop document upload
- Real-time query processing
- Query history and favorites

## 🔒 Security Considerations

- API key encryption and secure storage
- Input sanitization for all user inputs
- File type validation (PDF only)
- File size limits (max 50MB per file)
- CORS configuration
- Rate limiting

## 📊 Performance Optimization

- Async processing for multiple requests
- Document embedding caching
- Query result caching
- Efficient vector storage (optional)
- Connection pooling

## 🧪 Testing

```bash
# Run unit tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src
```

## 📈 Future Enhancements

- [ ] Vector database integration (Pinecone, Weaviate)
- [ ] Multi-language support
- [ ] Voice input/output
- [ ] Advanced table extraction
- [ ] Multi-modal support (images, charts)
- [ ] Collaborative features
- [ ] Export to various formats

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📧 Support

For questions or issues, please open a GitHub issue.

