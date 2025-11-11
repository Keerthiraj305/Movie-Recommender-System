# Movie Recommandation System

A small app that serves movie recommendations from precomputed pickles and fetches poster images from TMDB (The Movie Database).

## Quick summary

- App entry: `app.py`
- Model files (required): `model/movie_list.pkl` and `model/similarity.pkl` (already included in the `model/` directory)
- Health check: `GET /health` returns `OK` (200)

## Prerequisites

- Python 3.10+ (recommended)
- Git (optional)

## Install & run (Windows PowerShell)

Open PowerShell in the project directory and run:

```powershell
# create and activate a virtual environment
python -m venv .venv; .\.venv\Scripts\Activate.ps1

# install dependencies
pip install -r requirements.txt

# (optional) set TMDB API key for higher poster reliability
$env:TMDB_API_KEY = "YOUR_TMDB_API_KEY_HERE"

# run the app locally
python app.py
```

The app will start on port 5000 by default. Visit http://localhost:5000 in your browser.

Notes for Unix / production deployments:

- Use a process manager or WSGI server such as gunicorn: `gunicorn app:app` (Linux/macOS)
- The repo already contains a `Procfile` for Heroku/Render-like deployments.

## Environment variables

- `TMDB_API_KEY` (recommended) — your TMDB API key used to fetch poster images. If unset, poster calls will return None.
- `FLASK_SECRET_KEY` — optional; defaults to a development-only value. Set this in production.

## Healthcheck

After the app is running, you can verify it's healthy via PowerShell:

```powershell
Invoke-RestMethod -Uri http://localhost:5000/health
```

## Notes & suggestions

- The app loads two pickled model files at startup; ensure both `movie_list.pkl` and `similarity.pkl` remain in `model/`.
- Poster images are fetched live from TMDB on each recommendation. Consider caching poster URLs or adding a fallback image to reduce external requests and improve responsiveness.
- The repository has a default TMDB API key fallback in `app.py`. For security and reliability, override it with your own `TMDB_API_KEY` environment variable.

## License

See repository owner for licensing information.