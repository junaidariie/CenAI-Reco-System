from dotenv import load_dotenv
import os
import edge_tts
import tempfile
from uuid import uuid4
import requests
load_dotenv()


API_KEY = os.getenv("TMDB_API_KEY")
BASE_URL = "https://api.themoviedb.org/3"
IMAGE_BASE = "https://image.tmdb.org/t/p/w500"

#--------------------------------- get movie id ----------------------------
def get_movie_id(movie_title):
    url = f"{BASE_URL}/search/movie"
    params = {
        "api_key": API_KEY,
        "query": movie_title
    }

    r = requests.get(url, params=params)
    data = r.json()

    if "results" in data and len(data["results"]) > 0:
        return data["results"][0]["id"]
    return None

#--------------------------------- get movie reviews ----------------------------
def get_movie_reviews(movie_id, max_reviews=100):
    url = f"{BASE_URL}/movie/{movie_id}/reviews"
    params = {
        "api_key": API_KEY,
        "language": "en-US"
    }

    response = requests.get(url, params=params)

    if response.status_code != 200:
        print("TMDB Error:", response.status_code, response.text)
        return []

    data = response.json()

    reviews = []

    for review in data.get("results", [])[:max_reviews]:
        reviews.append({
            "author": review.get("author"),
            "content": review.get("content"),
            "rating": review.get("author_details", {}).get("rating"),
            "created_at": review.get("created_at")
        })

    return reviews

#--------------------------------- get full movie details ----------------------------
def get_movie_details(movie_id):
    url = f"{BASE_URL}/movie/{movie_id}"
    params = {"api_key": API_KEY}

    r = requests.get(url, params=params)
    data = r.json()

    details = {
        "title": data.get("title"),
        "overview": data.get("overview"),
        "release_date": data.get("release_date"),
        "runtime": data.get("runtime"),
        "rating": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "genres": [g["name"] for g in data.get("genres", [])],
        "poster": IMAGE_BASE + data["poster_path"] if data.get("poster_path") else None,
        "backdrop": IMAGE_BASE + data["backdrop_path"] if data.get("backdrop_path") else None
    }

    return details

#----------------------------------------- TTS ----------------------------------------------
async def TTS(text: str) -> str:
    if not text:
        return ""
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    temp_path = temp_file.name
    temp_file.close() 

    communicate = edge_tts.Communicate(text, "en-US-AriaNeural")
    await communicate.save(temp_path)

    return temp_path