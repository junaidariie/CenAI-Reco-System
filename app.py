import os
import requests
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Literal, List, Optional
from fastapi.middleware.cors import CORSMiddleware
from summarise_bot import summarise_movie as workflow
from prediction_helper import recommend 
from utils import (get_movie_id, get_movie_details, get_movie_reviews, TTS)
from ReviewSentiment import analyze_reviews_sentiment


TMDB_API_KEY = "4ca4d3c95de0c88528c2682781127d55"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

app = FastAPI(title='Movie Recommendation System', version='2.1')

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# this is for loading local movie titles for search autocomplete
try:
    movies_df = pd.read_csv('artifacts/cleaned_movie.csv') 
    ALL_MOVIE_TITLES = movies_df['title'].dropna().unique().tolist()
    print(f"✅ Loaded {len(ALL_MOVIE_TITLES)} movies for local search.")
except Exception as e:
    print(f"⚠️ Warning: Could not load local movie list. ({e})")
    ALL_MOVIE_TITLES = []


class RecomendationInput(BaseModel):
    movie_title: str
    engine: Literal["embedding", "tfidf", "hybrid"] = "embedding"
    top_k: int = 5

class MovieInfo(BaseModel):
    title: str
    overview: str
    release_date: str
    runtime: int | None
    rating: float
    vote_count: int
    genres: list[str]
    poster: str | None
    backdrop: str | None

class MovieReviews(BaseModel):
    title : str
    num_reviews : int = 50

class WorkflowInput(BaseModel):
    title: str
    overview: str


def fetch_tmdb(endpoint: str, params: dict = {}):
    params['api_key'] = TMDB_API_KEY
    url = f"{TMDB_BASE_URL}{endpoint}"
    response = requests.get(url, params=params)
    return response.json() if response.status_code == 200 else None

def format_tmdb_movies(results: list):
    formatted = []
    for m in results:
        formatted.append({
            "title": m.get("title"),
            "poster": f"{TMDB_IMAGE_BASE}{m.get('poster_path')}" if m.get('poster_path') else None,
            "rating": m.get("vote_average"),
            "release_date": m.get("release_date", "N/A"),
            "id": m.get("id"),
            "vote_count": m.get("vote_count")
        })
    return formatted

# ENDPOINTS
@app.get('/')
def status():
    return {'message': 'API is live', 'movies_loaded': len(ALL_MOVIE_TITLES)}


@app.get('/movies/trending')
def get_trending(time_window: str = "week"):
    data = fetch_tmdb(f"/trending/movie/{time_window}")
    if not data: return []
    return format_tmdb_movies(data.get("results", []))

@app.get('/movies/popular')
def get_popular():
    data = fetch_tmdb("/movie/popular")
    if not data: return []
    return format_tmdb_movies(data.get("results", []))


@app.get('/movies/search')
def search_movies(query: str = Query(..., min_length=2)):
    q = query.lower()
    matches = [t for t in ALL_MOVIE_TITLES if q in t.lower()][:10]
    return {"results": matches}


@app.post('/recomendation')
def recomendation(input_data : RecomendationInput):
    if input_data.movie_title not in ALL_MOVIE_TITLES:
        raise HTTPException(status_code=404, detail="Movie not found in local dataset")

    try:
        results = recommend(movie_title=input_data.movie_title, engine=input_data.engine, top_k=input_data.top_k)
        return {"results": results.to_dict(orient='records')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/movie-info/{title}', response_model=MovieInfo)
def movie_info(title: str):
    try:
        movie_id = get_movie_id(title)
        if not movie_id:
             raise HTTPException(status_code=404, detail="Movie not found on TMDB")
        return get_movie_details(movie_id=movie_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/movie-reviews-sentiment")
def movie_reviews_sentiment(input_data: MovieReviews):
    try:
        movie_id = get_movie_id(movie_title=input_data.title)
        reviews_data = get_movie_reviews(movie_id=movie_id, max_reviews=input_data.num_reviews)

        if not reviews_data:
            raise HTTPException(status_code=404, detail="No reviews found")

        review_texts = [r["content"] for r in reviews_data if r.get("content")]
        sentiment_distribution = analyze_reviews_sentiment(review_texts)

        return {
            "movie": input_data.title,
            "total_reviews_analyzed": len(review_texts),
            "sentiment_distribution": sentiment_distribution
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/TTS/{text}')
async def generate_tts(text: str):
    try:
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text is empty")
        audio_path = await TTS(text=text)

        if not os.path.exists(audio_path):
            raise HTTPException(status_code=500, detail="Audio generation failed")

        return FileResponse(audio_path, media_type="audio/mpeg", filename="summary_audio.mp3")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/summarize-movie')
def summarize_movie(input_data: WorkflowInput):
    try:
        return workflow(title=input_data.title, overview=input_data.overview)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))