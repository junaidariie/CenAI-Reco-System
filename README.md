# CineAI - Movie Recommendation System

A sophisticated movie recommendation system powered by multiple AI engines including embedding-based search, TF-IDF analysis, and hybrid approaches. Features real-time movie data, sentiment analysis, and AI-powered movie summaries.

## 🚀 Features

### Core Recommendation Engines
- **Embedding Engine (FAISS)**: Deep learning-based semantic similarity using Facebook AI Similarity Search
- **TF-IDF Engine**: Keyword-based recommendations using Term Frequency-Inverse Document Frequency
- **Hybrid Engine**: Combines both approaches with weighted scoring for optimal results

### AI-Powered Features
- **Movie Summarization**: LLM-generated insights, themes, and key moments
- **Sentiment Analysis**: Custom BiLSTM model for review sentiment classification
- **Text-to-Speech**: Audio summaries of movie analysis
- **Real-time Data**: Live movie information via TMDB API

### Interactive Web Interface
- Modern responsive design with dark theme
- Real-time search with autocomplete
- Interactive charts for sentiment visualization
- Audio playback for movie summaries

## 📋 Prerequisites

- Python 3.8+
- Node.js (for frontend dependencies)
- TMDB API Key
- OpenAI API Key (for LLM features)
- Groq API Key (alternative LLM provider)

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd final_reco_system
```

2. **Install Python dependencies**
```bash
pip install -r requirements.txt
```

3. **Set up environment variables**
Create a `.env` file in the root directory:
```env
TMDB_API_KEY=your_tmdb_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

4. **Download required model files**
Ensure these files are in the `artifacts/` directory:
- `cleaned_movie.csv` - Movie dataset
- `movie_embeddings.npy` - Pre-computed embeddings
- `movie_faiss.index` - FAISS index file
- `tfidf_vectorizer.pkl` - TF-IDF vectorizer
- `tfidf_matrix.pkl` - TF-IDF matrix
- `goemotions_bilstm_checkpoint.pth` - Sentiment analysis model

## 🚀 Quick Start

1. **Start the FastAPI server**
```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

2. **Open the web interface**
Navigate to `http://localhost:8000` and open `index.html` in your browser.

3. **Configure API endpoint**
Update the `API_BASE` variable in `index.html`:
```javascript
const API_BASE = "http://localhost:8000";
```

## 📚 API Documentation

### Core Endpoints

#### Get Recommendations
```http
POST /recomendation
Content-Type: application/json

{
    "movie_title": "Inception",
    "engine": "hybrid",
    "top_k": 5
}
```

#### Movie Information
```http
GET /movie-info/{title}
```

#### Sentiment Analysis
```http
POST /movie-reviews-sentiment
Content-Type: application/json

{
    "title": "Inception",
    "num_reviews": 50
}
```

#### AI Movie Summary
```http
POST /summarize-movie
Content-Type: application/json

{
    "title": "Inception",
    "overview": "A thief who steals corporate secrets..."
}
```

#### Text-to-Speech
```http
GET /TTS/{text}
```

### Search & Discovery

#### Search Movies
```http
GET /movies/search?query=inception
```

#### Trending Movies
```http
GET /movies/trending?time_window=week
```

#### Popular Movies
```http
GET /movies/popular
```

## 🏗️ Architecture

### Backend Components

- **FastAPI Application** (`app.py`): Main API server with CORS support
- **Recommendation Engine** (`prediction_helper.py`): Core ML algorithms
- **Sentiment Analysis** (`ReviewSentiment.py`): Custom BiLSTM model
- **Movie Summarization** (`summarise_bot.py`): LangGraph workflow for AI analysis
- **Utilities** (`utils.py`): TMDB API integration and TTS functionality

### Frontend Components

- **Modern Web Interface** (`index.html`): Responsive SPA with dark theme
- **Real-time Search**: Autocomplete with debounced API calls
- **Interactive Charts**: Sentiment visualization using Chart.js
- **Audio Integration**: Built-in audio player for summaries

### Machine Learning Models

1. **Embedding Model**: SentenceTransformer (all-MiniLM-L6-v2)
2. **FAISS Index**: Optimized vector similarity search
3. **TF-IDF Vectorizer**: Scikit-learn implementation
4. **Sentiment Model**: Custom BiLSTM trained on GoEmotions dataset

## 🔧 Configuration

### Recommendation Engine Settings

```python
# Hybrid engine weights
alpha = 0.6  # Embedding weight
beta = 0.4   # TF-IDF weight

# Final score calculation
final_score = (alpha * embedding_score) + (beta * tfidf_score)
```

### API Configuration

```python
# TMDB API settings
TMDB_API_KEY = "your_api_key"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"
```

## 📊 Model Performance

### Recommendation Engines Comparison

| Engine | Strengths | Use Cases |
|--------|-----------|-----------|
| **Embedding** | Semantic understanding, context-aware | Similar themes, plot elements |
| **TF-IDF** | Keyword matching, fast computation | Genre-based, cast/crew similarity |
| **Hybrid** | Best of both worlds, balanced results | General recommendations |

### Sentiment Analysis Metrics

- **Model**: BiLSTM with attention mechanism
- **Dataset**: GoEmotions (28 emotion categories)
- **Accuracy**: Optimized for movie review sentiment
- **Categories**: Positive, Negative, Neutral

## 🎯 Usage Examples

### Basic Movie Recommendation
```python
from prediction_helper import recommend

# Get hybrid recommendations
results = recommend(
    movie_title="The Dark Knight",
    engine="hybrid",
    top_k=5
)
```

### Sentiment Analysis
```python
from ReviewSentiment import analyze_reviews_sentiment

reviews = ["Great movie!", "Terrible plot", "Average film"]
sentiment = analyze_reviews_sentiment(reviews)
# Returns: {"positive": 33.33, "negative": 33.33, "neutral": 33.33}
```

### AI Movie Summary
```python
from summarise_bot import summarise_movie

summary = summarise_movie(
    title="Inception",
    overview="A thief who steals corporate secrets..."
)
```

## 🔍 Troubleshooting

### Common Issues

1. **Movie not found in recommendations**
   - Ensure movie exists in `cleaned_movie.csv`
   - Check exact title spelling

2. **FAISS index errors**
   - Verify `movie_faiss.index` file integrity
   - Ensure embeddings dimension matches (384)

3. **API rate limits**
   - TMDB API has rate limits (40 requests/10 seconds)
   - Implement caching for production use

4. **Memory issues**
   - Large models require sufficient RAM
   - Consider model quantization for deployment

## 🚀 Deployment

### Production Considerations

1. **Environment Setup**
```bash
# Use production WSGI server
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker
```

2. **Caching Strategy**
```python
# Implement Redis caching for API responses
# Cache movie data, recommendations, and sentiment results
```

3. **Model Optimization**
```python
# Consider model quantization
# Implement batch processing for recommendations
# Use async processing for heavy computations
```

## 📈 Performance Optimization

- **FAISS Index**: Optimized for fast similarity search
- **Async Processing**: Non-blocking API operations
- **Caching**: Reduce API calls and computation
- **Batch Processing**: Handle multiple requests efficiently

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **TMDB**: Movie data and images
- **FAISS**: Efficient similarity search
- **GoEmotions**: Sentiment analysis dataset
- **SentenceTransformers**: Embedding models
- **LangChain**: LLM orchestration framework

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the troubleshooting section
- Review API documentation

---

**Built with ❤️ using FastAPI, FAISS, and modern web technologies**