import os
import pickle
import requests
from flask import Flask, render_template, request, abort

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "change_this_secret_for_prod")

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
MOVIE_PKL = os.path.join(MODEL_DIR, "movie_list.pkl")
SIM_PKL = os.path.join(MODEL_DIR, "similarity.pkl")

# TMDB API key from env (recommended). Fallback kept for backwards compatibility,
# but you should set TMDB_API_KEY in Render dashboard.
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8")

# Load data (fail fast with helpful message)
try:
    movies = pickle.load(open(MOVIE_PKL, "rb"))
    similarity = pickle.load(open(SIM_PKL, "rb"))
except FileNotFoundError as e:
    # If model files are missing, raise a helpful error at startup
    raise RuntimeError(
        f"Missing model files. Ensure `{MOVIE_PKL}` and `{SIM_PKL}` exist in your repo."
    ) from e
except Exception as e:
    raise RuntimeError(f"Error loading pickles: {e}") from e


def fetch_poster(movie_id):
    """Return full poster URL from TMDB for a given movie_id, or None on failure."""
    if not TMDB_API_KEY:
        return None
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}&language=en-US"
        r = requests.get(url, timeout=6)
        r.raise_for_status()
        data = r.json()
        poster_path = data.get("poster_path")
        if not poster_path:
            return None
        return "https://image.tmdb.org/t/p/w500/" + poster_path
    except Exception:
        return None


def recommend(movie_title, top_k=5):
    """Return (titles_list, posters_list) for top_k recommendations."""
    try:
        index = movies[movies['title'] == movie_title].index[0]
    except Exception:
        return [], []

    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])
    recommended_movie_names = []
    recommended_movie_posters = []

    for i in distances[1: top_k+1]:
        movie_idx = i[0]
        recommended_movie_names.append(movies.iloc[movie_idx].title)
        movie_id = movies.iloc[movie_idx].movie_id
        poster_url = fetch_poster(movie_id)
        recommended_movie_posters.append(poster_url)

    return recommended_movie_names, recommended_movie_posters


@app.route("/", methods=["GET", "POST"])
def index():
    movie_list = list(movies['title'].values)
    recommendations = []
    selected_movie = None

    if request.method == "POST":
        selected_movie = request.form.get("movie")
        if selected_movie:
            names, posters = recommend(selected_movie, top_k=5)
            recommendations = list(zip(names, posters))

    return render_template(
        "index.html",
        movie_list=movie_list,
        recommendations=recommendations,
        selected_movie=selected_movie
    )


# Helpful healthcheck route
@app.route("/health")
def health():
    return "OK", 200


# Only used when running `python app.py` locally
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
