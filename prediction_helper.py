import numpy as np
import pandas as pd
import faiss
import pickle
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings("ignore", module="sklearn")

df = pd.read_csv("artifacts/cleaned_movie.csv")

# --------------------- loading models -------------------------
print("LOADING THE MODELS...")

embeddings = np.load("artifacts/movie_embeddings.npy")
print("Embeddings shape:", embeddings.shape)

index = faiss.read_index("artifacts/movie_faiss.index")
print("FAISS index loaded. Total vectors:", index.ntotal)

with open("artifacts/tfidf_vectorizer.pkl", "rb") as f:
    tfidf_vectorizer = pickle.load(f)
print("tfidf_vectorizer loaded.")

with open("artifacts/tfidf_matrix.pkl", "rb") as f:
    tfidf_matrix = pickle.load(f)
print("tfidf_matrix loaded")

model = SentenceTransformer("all-MiniLM-L6-v2")
print("SentenceTransformer loaded.")

print("ALL MODELS LOADED SUCCESFULLY.")


#------------------------------ loading engines ---------------------------
def recommend_movies(movie_title, df, model, index, top_k=10):
    try:
        if movie_title not in df['title'].values:
            return f"Movie '{movie_title}' not found in dataset."
        idx = df[df['title'] == movie_title].index[0]
        query_text = df.loc[idx, 'tags']
        query_embedding = model.encode([query_text])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        scores, indices = index.search(query_embedding, top_k + 1)
        sim_scores = scores[0][1:]
        sim_indices = indices[0][1:]

        results = df.iloc[sim_indices].copy()
        results["embedding_score"] = sim_scores

        return results[['title', 'embedding_score']]
    except Exception as e:
        raise Exception(f"Error while recomending movies [embeddings] : {e}")

def recommend_movies_tfidf(movie_title, df, tfidf_matrix, top_k=5):
    try:
        if movie_title not in df['title'].values:
            return f"Movie '{movie_title}' not found in dataset."

        idx = df[df['title'] == movie_title].index[0]

        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

        sim_indices = cosine_sim.argsort()[::-1][1:top_k+1]

        results = df.iloc[sim_indices].copy()
        results["tfidf_score"] = cosine_sim[sim_indices]

        return results[['title', 'tfidf_score']]
    except Exception as e:
        raise Exception(f"Error while recomending movies [tfidf] : {e}")
    

def recommend_movies_hybrid(movie_title, df, model, index, tfidf_matrix, top_k=10, alpha=0.6):
    try:
        """
        alpha = weight for embedding score
        (1 - alpha) = weight for tf-idf score
        """

        if movie_title not in df['title'].values:
            return f"Movie '{movie_title}' not found in dataset."

        idx = df[df['title'] == movie_title].index[0]
        query_text = df.loc[idx, 'tags']

        # -------- Embedding Search --------
        query_embedding = model.encode([query_text])
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)

        emb_scores, emb_indices = index.search(query_embedding, 50)
        emb_scores = emb_scores[0]
        emb_indices = emb_indices[0]

        emb_df = pd.DataFrame({
            "index": emb_indices,
            "embedding_score": emb_scores
        })

        # -------- TF-IDF Search --------
        cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        tfidf_indices = cosine_sim.argsort()[::-1][:50]
        tfidf_scores = cosine_sim[tfidf_indices]

        tfidf_df = pd.DataFrame({
            "index": tfidf_indices,
            "tfidf_score": tfidf_scores
        })

        # -------- Merge Both --------
        merged = pd.merge(emb_df, tfidf_df, on="index", how="outer").fillna(0)

        # -------- Normalize Scores --------
        merged["embedding_score"] = merged["embedding_score"] / merged["embedding_score"].max()
        merged["tfidf_score"] = merged["tfidf_score"] / merged["tfidf_score"].max()

        # -------- Weighted Fusion --------
        merged["hybrid_score"] = alpha * merged["embedding_score"] + (1 - alpha) * merged["tfidf_score"]

        # -------- Final Ranking --------
        merged = merged.sort_values(by="hybrid_score", ascending=False)

        top_indices = merged["index"].head(top_k).values
        results = df.iloc[top_indices].copy()
        results["hybrid_score"] = merged["hybrid_score"].head(top_k).values

        return results[['title', 'hybrid_score']]
    except Exception as e:
        raise Exception(f"Error while recomending movies [hybrid] : {e}")

def recommend(movie_title, engine="embedding", top_k=5):
    if engine == "embedding":
        return recommend_movies(movie_title, df, model, index, top_k)

    elif engine == "tfidf":
        return recommend_movies_tfidf(movie_title, df, tfidf_matrix, top_k)

    elif engine == "hybrid":
        return recommend_movies_hybrid(movie_title, df, model, index, tfidf_matrix, top_k)

    else:
        return "Invalid engine. Choose: 'embedding', 'tfidf', or 'hybrid'."


# ----------------------------- testing ---------------------------------------
"""print(recommend("Toy Story", engine="embedding", top_k=5))
print(recommend("Toy Story", engine="tfidf", top_k=5))
print(recommend("Toy Story", engine="hybrid", top_k=5))"""