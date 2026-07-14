# Double-Slit Simulation

## Minimal web deployment

This app is already a Streamlit site. To host it and share a link:

1. Push the repo to GitHub.
2. Deploy it on Streamlit Community Cloud or Hugging Face Spaces.
3. Set the app entry file to `app.py`.

### Files needed for web hosting

- `app.py` — root entrypoint that points Streamlit at `src/main.py`
- `requirements.txt` — runtime deps only: `streamlit`, `numpy`, `matplotlib`
- `src/` — the simulation code

### Why not GitHub Pages?

GitHub Pages only serves static files. This simulation needs Python on the server, so use Streamlit Cloud or Hugging Face Spaces instead.