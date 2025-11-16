# NIT Kurukshetra RAG System

A complete web scraping and Retrieval-Augmented Generation (RAG) system for the NIT Kurukshetra website. This system scrapes the website, extracts text content, generates vector embeddings, and provides an interactive question-answering interface powered by Groq LLM.

## Features

- 🕷️ **Web Scraping**: Recursively scrapes NIT KKR website with BeautifulSoup
- 📝 **Text Extraction**: Clean text extraction from HTML pages
- 🧠 **Vector Embeddings**: Generate embeddings using sentence-transformers
- 🔍 **Similarity Search**: FAISS-based vector similarity search
- 🤖 **RAG System**: Interactive question-answering interface
- 🧠 **Groq LLM Integration**: Powered by Groq's fast language models
- 📊 **Statistics**: Comprehensive system statistics and monitoring

## Installation

1. Clone or download this project
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. **Configure Groq API (Optional but Recommended)**:
   - Get your free API key from [Groq Console](https://console.groq.com/keys)
   - Run the setup script:
   ```bash
   python setup_groq.py
   ```
   - Or manually set environment variable:
   ```bash
   set GROQ_API_KEY=your_api_key_here  # Windows
   export GROQ_API_KEY=your_api_key_here  # Linux/Mac
   ```

**Note**: Without Groq API key, the system will use template-based responses. With Groq, you get much more intelligent and contextual answers.

## Usage

### Quick Start (Complete Pipeline)

Run the complete pipeline in one command:

```bash
python main.py full
```

This will:
1. Scrape the NIT KKR website
2. Generate vector embeddings
3. Start the interactive RAG system

### Individual Commands

#### 1. Web Scraping

```bash
python main.py scrape
```

- Scrapes the NIT KKR website recursively (depth=2)
- Saves HTML pages to `nitkkr_pages/`
- Extracts text content to `extracted_text/`
- Includes respectful delays between requests

#### 2. Vector Embeddings

```bash
python main.py embed
```

- Generates embeddings using `all-MiniLM-L6-v2` model
- Creates text chunks with overlap for better context
- Builds FAISS index for similarity search
- Saves vector store to `vector_store/`

#### 3. RAG System

```bash
python main.py rag
```

- Starts interactive question-answering interface
- Supports natural language queries about NIT KKR
- Provides source citations and relevance scores
- **With Groq**: Intelligent, contextual responses
- **Without Groq**: Template-based responses

#### 4. System Statistics

```bash
python main.py stats
```

- Shows system statistics and metrics
- Displays total chunks, documents, and word counts

## Project Structure

```
NIT_KKR_project/
├── main.py                 # Main orchestration script
├── scraper.py              # Web scraping module
├── vector_embeddings.py    # Vector embedding generation
├── rag_system.py           # RAG system implementation
├── groq_llm.py            # Groq LLM service integration
├── config.py              # Configuration management
├── setup_groq.py          # Groq API setup script
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── .env                   # Environment variables (create with setup_groq.py)
├── nitkkr_pages/          # Scraped HTML pages
├── extracted_text/         # Clean text files
├── vector_store/           # FAISS index and metadata
└── embeddings/             # Embedding files
```

## Configuration

### Groq LLM Settings

In `config.py`, you can modify:
- `GROQ_MODEL`: Available models: `llama3-8b-8192`, `llama3-70b-8192`, `mixtral-8x7b-32768`
- `MAX_RESPONSE_TOKENS`: Maximum tokens for LLM responses (default: 1024)

### Scraper Settings

In `scraper.py`, you can modify:
- `max_depth`: Maximum crawling depth (default: 2)
- `delay`: Delay between requests in seconds (default: 1)

### Embedding Settings

In `vector_embeddings.py`, you can modify:
- `model_name`: Sentence transformer model (default: "all-MiniLM-L6-v2")
- `chunk_size`: Text chunk size (default: 500)
- `chunk_overlap`: Overlap between chunks (default: 50)

## Example Queries

The RAG system can answer questions like:

- "What are the admission requirements for NIT KKR?"
- "Tell me about the computer science department"
- "What facilities are available on campus?"
- "How can I apply for courses?"
- "What are the contact details?"
- "Tell me about the faculty"

## Technical Details

### Web Scraping
- Uses BeautifulSoup for HTML parsing
- Implements respectful crawling with delays
- Handles various content types and errors gracefully
- Extracts clean text by removing navigation, scripts, and styles

### Vector Embeddings
- Uses sentence-transformers for high-quality embeddings
- Implements text chunking for better context preservation
- Normalizes embeddings for cosine similarity
- Stores metadata for source tracking

### RAG System
- Implements retrieval-augmented generation
- Provides source citations and relevance scores
- Supports interactive querying
- Includes error handling and graceful degradation
- **Groq Integration**: Uses Groq's fast LLMs for intelligent response generation
- **Fallback Support**: Gracefully falls back to template-based responses if Groq is unavailable

## Dependencies

- `requests`: HTTP requests for web scraping
- `beautifulsoup4`: HTML parsing and text extraction
- `sentence-transformers`: Vector embedding generation
- `faiss-cpu`: Vector similarity search
- `numpy`: Numerical operations
- `urllib3`: URL parsing and handling
- `groq`: Groq LLM API integration
- `python-dotenv`: Environment variable management

## Troubleshooting

### Common Issues

1. **"Vector store not found"**
   - Run `python main.py embed` first to generate embeddings

2. **"No scraped data found"**
   - Run `python main.py scrape` first to scrape the website

3. **Import errors**
   - Install dependencies: `pip install -r requirements.txt`

4. **Groq API errors**
   - Check your API key: `python setup_groq.py`
   - Verify API key at [Groq Console](https://console.groq.com/keys)
   - System will fallback to template responses if Groq fails

5. **Network errors during scraping**
   - Check internet connection
   - The scraper includes retry logic and error handling

### Performance Tips

- For faster scraping, reduce the delay in `scraper.py`
- For better search results, increase chunk overlap in `vector_embeddings.py`
- For more comprehensive coverage, increase max_depth in scraping
- **Groq Models**: Use `llama3-8b-8192` for speed, `llama3-70b-8192` for quality
- **Response Quality**: Adjust `MAX_RESPONSE_TOKENS` in `config.py` for longer responses

## License

This project is for educational purposes. Please respect the NIT Kurukshetra website's robots.txt and terms of service when scraping.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.
