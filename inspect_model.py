import os
import pickle
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
MOVIE_PKL = os.path.join(MODEL_DIR, "movie_list.pkl")
SIM_PKL = os.path.join(MODEL_DIR, "similarity.pkl")

print('Loading pickles...')
movies = pickle.load(open(MOVIE_PKL, 'rb'))
similarity = pickle.load(open(SIM_PKL, 'rb'))

print('\n--- Movies object ---')
print(type(movies))
try:
    import pandas as pd
    if isinstance(movies, pd.DataFrame):
        print('DataFrame shape:', movies.shape)
        print('Columns:', list(movies.columns))
        print('Dtypes:\n', movies.dtypes)
        print('\nHead:\n', movies.head().to_string())
    else:
        print(repr(movies)[:1000])
except Exception as e:
    print('Could not inspect movies as DataFrame:', e)

print('\n--- Similarity object ---')
print(type(similarity))
try:
    sim_arr = np.array(similarity)
    print('Similarity shape:', sim_arr.shape)
    print('Similarity dtype:', sim_arr.dtype)
    print('Similarity stats: mean={:.4f}, median={:.4f}, min={:.4f}, max={:.4f}'.format(
        float(np.nanmean(sim_arr)), float(np.nanmedian(sim_arr)), float(np.nanmin(sim_arr)), float(np.nanmax(sim_arr))
    ))
except Exception as e:
    print('Could not convert similarity to array:', e)

# Try proxy evaluation if genre-like column exists
print('\n--- Attempting proxy evaluation (genre overlap) ---')
try:
    import pandas as pd
    if not isinstance(movies, pd.DataFrame):
        print('movies is not a DataFrame; skipping proxy evaluation')
    else:
        # find possible genre columns
        genre_cols = [c for c in movies.columns if 'genre' in c.lower() or 'category' in c.lower() or 'tags' in c.lower()]
        print('Candidate genre-like columns:', genre_cols)
        if not genre_cols:
            print('No genre-like column found; skipping proxy evaluation')
        else:
            col = genre_cols[0]
            print('Using column:', col)
            # Normalize genres into sets
            def parse_genres(x):
                if pd.isna(x):
                    return set()
                if isinstance(x, (list, set, tuple)):
                    return set(x)
                s = str(x)
                # common delimiters
                for d in ['|', ',', ';']:
                    if d in s:
                        parts = [p.strip() for p in s.split(d) if p.strip()]
                        return set(parts)
                # fallback: single token
                return set([s.strip()])

            genres = movies[col].apply(parse_genres).tolist()
            n = min(500, len(movies))
            K_list = [5, 10]
            results = {}
            sim_arr = np.array(similarity)
            for K in K_list:
                precisions = []
                for i in range(n):
                    # get top K similar excluding self
                    sims = list(enumerate(sim_arr[i]))
                    sims_sorted = sorted(sims, key=lambda x: x[1], reverse=True)
                    topk = [idx for idx,_ in sims_sorted[1:K+1]]
                    gt = genres[i]
                    if not gt:
                        continue
                    match_count = 0
                    for j in topk:
                        if genres[j] & gt:
                            match_count += 1
                    precisions.append(match_count / float(K))
                if precisions:
                    results[K] = float(np.mean(precisions))
                else:
                    results[K] = None
            print('Proxy precision@K (genre overlap) for first', n, 'movies:')
            for K in K_list:
                print(f'  precision@{K}: {results[K]}')
except Exception as e:
    print('Proxy evaluation failed:', e)

print('\nDone.')
