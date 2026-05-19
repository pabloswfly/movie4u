# movie4u 🎬

A content-based movie recommender that learns your taste from a list of films you've seen and suggests new ones to watch.

**Live app → [movie4u.streamlit.app](https://movie4u.streamlit.app)**

---

## How it works

1. **You provide a rated movie list** — one film per line, with optional year and rating out of 10.
2. **Each film is matched on TMDB** to retrieve its synopsis, genres, directors, cast, and keywords.
3. **Synopsis embeddings** are computed with `all-MiniLM-L6-v2` (sentence-transformers). Dimensions that co-vary with your ratings are amplified via Pearson correlation weighting.
4. **A taste vector** is built as the weighted sum of your rated embeddings (weight = rating − 5), so films you loved pull it in one direction and films you disliked push it away.
5. **A large TMDB catalog** (up to 20 000 movies, across vote-count and popularity rankings) is encoded and ranked by cosine similarity against your taste vector.
6. **A categorical bonus** (genres, directors, keywords) contributes 20 % of the final score.
7. **An interactive 2D scatter plot** (UMAP projection) lets you explore where recommendations sit relative to your rated films.

## Input format

```
The Godfather (1972): 10
Blade Runner (1982): 9
Mulholland Drive (2001): 8
The Big Lebowski: 7
Interstellar: 6
```

Each line accepts `Title (Year): Rating`, `Title (Year) - Rating`, `Title: Rating`, or just `Title`. Ratings default to 10 if omitted.

## Running locally

```bash
# 1. Clone
git clone https://github.com/pabloswfly/movie4u.git
cd movie4u

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your TMDB API key (free at themoviedb.org)
echo "TMDB_API_KEY=your_key_here" > .env

# 4. Launch
streamlit run main.py
```

Or enter the API key directly in the sidebar when the app opens.

## Tech stack

| Layer | Library |
|---|---|
| UI | Streamlit |
| Embeddings | sentence-transformers (`all-MiniLM-L6-v2`) |
| Dimensionality reduction | UMAP / PCA (scikit-learn fallback) |
| Interactive plot | Plotly |
| Movie data | TMDB API |
