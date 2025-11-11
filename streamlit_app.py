import os
import pickle
import requests
import streamlit as st
import pandas as pd
import io
import numpy as np

# Paths
BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
MOVIE_PKL = os.path.join(MODEL_DIR, "movie_list.pkl")
SIM_PKL = os.path.join(MODEL_DIR, "similarity.pkl")

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "8265bd1679663a7ea12ac168da84d2e8")

# Load data
@st.cache_data
def load_data():
    try:
        movies = pickle.load(open(MOVIE_PKL, "rb"))
        similarity = pickle.load(open(SIM_PKL, "rb"))
    except FileNotFoundError as e:
        st.error(f"Missing model files. Ensure '{MOVIE_PKL}' and '{SIM_PKL}' exist.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.stop()
    return movies, similarity


@st.cache_data
def fetch_poster(movie_id):
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


def recommend(movie_title, movies, similarity, top_k=5):
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


def main():
    st.set_page_config(page_title="Movie Recommender", layout="wide")
    st.title("Movie Recommender System")

    movies, similarity = load_data()
    movie_list = list(movies['title'].values)
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Recommendations", "Similarity Matrix"]) 

    if page == "Recommendations":
        selected = st.selectbox("Select a movie", options=movie_list)
        if st.button("Show Recommendations"):
            names, posters = recommend(selected, movies, similarity, top_k=5)
            cols = st.columns(len(names))
            for c, name, poster in zip(cols, names, posters):
                with c:
                    if poster:
                        st.image(poster, width='stretch', caption=name)
                    else:
                        st.write("No image")
                        st.write(name)

    else:
        # Similarity Matrix page
        st.header("Similarity Matrix")
        max_show = min(200, len(movies))
        n = st.sidebar.slider("Number of movies to show (subset)", min_value=10, max_value=max_show, value=min(50, max_show), step=5)

        # Build subset dataframe
        try:
            subset_titles = movies['title'].iloc[:n].values
            df_subset = pd.DataFrame(similarity[:n, :n], index=subset_titles, columns=subset_titles)
        except Exception:
            # If similarity is not a numpy array, try to convert
            sim_arr = pd.DataFrame(similarity)
            df_subset = sim_arr.iloc[:n, :n]

        st.markdown(f"Displaying a {n}x{n} subset of the similarity matrix. Use the slider to change the size.")
        st.dataframe(df_subset)

        # Download button for the subset
        csv = df_subset.to_csv(index=True).encode('utf-8')
        st.download_button("Download CSV (subset)", data=csv, file_name=f"similarity_subset_{n}x{n}.csv", mime='text/csv')

        # Optional evaluation page (proxy metrics using tag overlap)
        if st.sidebar.checkbox("Show Evaluation Page"):
            st.header("Model Evaluation (proxy)")
            st.markdown("This uses the `tags` column as a proxy for relevance. Metrics are heuristic and for exploration only.")

            # evaluation controls
            max_eval = min(1000, len(movies))
            n_eval = st.slider("Number of movies to evaluate (subset)", min_value=50, max_value=max_eval, value=200, step=50)
            K_eval = st.select_slider("K for precision/recall/F1", options=[5,10,20,50], value=5)

            # prepare tag sets
            def parse_tags(x):
                if pd.isna(x):
                    return set()
                if isinstance(x, (list, set, tuple)):
                    return set(x)
                s = str(x)
                for d in ['|', ',', ';']:
                    if d in s:
                        parts = [p.strip() for p in s.split(d) if p.strip()]
                        return set(parts)
                return set([s.strip()])

            tag_col_candidates = [c for c in movies.columns if 'tag' in c.lower() or 'genre' in c.lower() or 'category' in c.lower()]
            if not tag_col_candidates:
                st.warning('No tag/genre-like column available for proxy evaluation.')
            else:
                col = tag_col_candidates[0]
                st.write('Using column for proxy relevance:', col)
                tags = movies[col].apply(parse_tags).tolist()

                sim_arr = np.array(similarity)
                precisions = []
                recalls = []
                f1s = []
                for i in range(min(n_eval, len(movies))):
                    gt = tags[i]
                    if not gt:
                        continue
                    sims = list(enumerate(sim_arr[i]))
                    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
                    topk = [idx for idx,_ in sims_sorted[1:K_eval+1]]
                    relevant_set = set(j for j in range(len(movies)) if j!=i and (tags[j] & gt))
                    if not relevant_set:
                        continue
                    tp = sum(1 for j in topk if j in relevant_set)
                    prec = tp / float(K_eval)
                    rec = tp / float(len(relevant_set))
                    if prec+rec > 0:
                        f1 = 2*prec*rec/(prec+rec)
                    else:
                        f1 = 0.0
                    precisions.append(prec)
                    recalls.append(rec)
                    f1s.append(f1)

                if precisions:
                    import statistics
                    st.metric('Avg precision@K', f"{statistics.mean(precisions):.4f}")
                    st.metric('Avg recall@K', f"{statistics.mean(recalls):.4f}")
                    st.metric('Avg F1@K', f"{statistics.mean(f1s):.4f}")

                    eval_df = pd.DataFrame({'precision': precisions, 'recall': recalls, 'f1': f1s})
                    st.dataframe(eval_df.describe())
                    csv_eval = eval_df.to_csv(index=False).encode('utf-8')
                    st.download_button('Download evaluation CSV', data=csv_eval, file_name=f'eval_proxy_{n_eval}_K{K_eval}.csv', mime='text/csv')


if __name__ == '__main__':
    main()
