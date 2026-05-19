import time
from collections import defaultdict

import numpy as np

from features import encode
from tmdb_client import TMDBClient

_SORT_ORDERS = ["vote_count.desc", "popularity.desc"]
_TMDB_MAX_PAGES = 500


def enrich_with_tmdb(
    ratings: list[dict],
    client: TMDBClient,
    progress_callback=None,
) -> tuple[list[dict], list[str]]:
    """Match each rated movie to a TMDB entry and attach metadata.

    Returns (enriched, not_found) where not_found is a list of titles
    that could not be matched on TMDB.
    """
    enriched: list[dict] = []
    not_found: list[str] = []

    for i, movie in enumerate(ratings):
        result = client.search_movie(movie["title"], movie.get("year"))
        if not result:
            not_found.append(movie["title"])
            if progress_callback:
                progress_callback(i + 1, len(ratings))
            continue

        try:
            details = client.get_movie_details(result["id"])
        except Exception:
            not_found.append(movie["title"])
            if progress_callback:
                progress_callback(i + 1, len(ratings))
            continue

        enriched.append(
            {
                **movie,
                "tmdb_id": result["id"],
                "poster_path": details.get("poster_path"),
                "overview": details.get("overview") or "",
                "tmdb_rating": details.get("vote_average", 0),
                **client.extract_features(details),
            }
        )

        if progress_callback:
            progress_callback(i + 1, len(ratings))

        time.sleep(0.1)

    return enriched, not_found


def _dimension_weights(embeddings: np.ndarray, ratings: np.ndarray) -> np.ndarray:
    """Per-dimension weights derived from Pearson correlation with user ratings.

    Dimensions that co-vary with what the user likes are amplified; dimensions
    with no predictive signal stay at weight 1.  sqrt smooths extreme values so
    a handful of highly-correlated dimensions don't dominate.  The result is
    normalised so the mean weight is 1, preserving the overall similarity scale.
    """
    r = ratings - ratings.mean()
    e = embeddings - embeddings.mean(axis=0)
    std_r = r.std() + 1e-8
    std_e = e.std(axis=0).clip(1e-8)

    corr = (e * r[:, np.newaxis]).mean(axis=0) / (std_e * std_r)  # (D,)
    weights = np.sqrt(np.abs(corr))

    mean_w = weights.mean()
    return weights / mean_w if mean_w > 0 else np.ones_like(weights)


def _build_categorical_profile(enriched: list[dict]) -> dict[str, float]:
    """Weighted genre/director/keyword profile: weight = (rating - 5)."""
    profile: dict[str, float] = defaultdict(float)
    for movie in enriched:
        w = movie["user_rating"] - 5
        for genre in movie.get("genres", []):
            profile[f"genre:{genre}"] += w
        for director in movie.get("directors", []):
            profile[f"director:{director}"] += w * 1.5
        for kw in movie.get("keywords", []):
            profile[f"kw:{kw}"] += w
    return dict(profile)


def _categorical_bonus(movie: dict, profile: dict[str, float]) -> float:
    score = 0.0
    for genre in movie.get("genres", []):
        score += profile.get(f"genre:{genre}", 0)
    for director in movie.get("directors", []):
        score += profile.get(f"director:{director}", 0) * 1.5
    for kw in movie.get("keywords", []):
        score += profile.get(f"kw:{kw}", 0)
    return score


def _apply_weights_and_normalise(normed: np.ndarray, dim_w: np.ndarray) -> np.ndarray:
    """Scale embedding dimensions by dim_w then re-normalise each row."""
    scaled = normed * dim_w
    norms = np.linalg.norm(scaled, axis=1, keepdims=True)
    return scaled / np.where(norms > 0, norms, 1.0)


def get_recommendations(
    enriched: list[dict],
    client: TMDBClient,
    n: int = 20,
    catalog_size: int = 1000,
    progress_callback=None,
) -> list[dict]:
    """Return top-N recommendations using weighted cosine similarity.

    Scoring pipeline:
      1. Compute per-dimension weights from the Pearson correlation between
         each embedding dimension and the user's ratings.  This focuses
         similarity on the aspects of synopses that actually predict taste.
      2. Build a taste vector: weighted sum of dimension-scaled rated
         embeddings (weight = user_rating - 5), then normalise.
      3. Paginate TMDB discover (two sort orders) for a large candidate
         catalog; batch-encode all overviews and rank by cosine similarity.
      4. Fetch full details for the top candidates; add a categorical bonus
         (genres / directors / keywords) worth 20% of the final score.
    """
    # 1. Encode rated synopses and compute dimension weights
    rated_embeddings = encode([m.get("overview") or "" for m in enriched])
    ratings = np.array([m["user_rating"] for m in enriched], dtype=float)

    norms = np.linalg.norm(rated_embeddings, axis=1, keepdims=True)
    normed_rated = rated_embeddings / np.where(norms > 0, norms, 1.0)

    dim_w = _dimension_weights(rated_embeddings, ratings)
    normed_rated_w = _apply_weights_and_normalise(normed_rated, dim_w)

    # 2. Build taste vector
    rating_weights = ratings - 5                                   # range −4 … +5
    taste_vector = (normed_rated_w * rating_weights[:, np.newaxis]).sum(axis=0)
    taste_norm = np.linalg.norm(taste_vector)
    if taste_norm > 0:
        taste_vector /= taste_norm

    profile = _build_categorical_profile(enriched)
    seen_ids = {m["tmdb_id"] for m in enriched}

    # 3. Paginate TMDB discover catalog across two sort orders
    pages_per_order = min(
        (catalog_size // 20 + len(_SORT_ORDERS) - 1) // len(_SORT_ORDERS),
        _TMDB_MAX_PAGES,
    )
    total_pages = pages_per_order * len(_SORT_ORDERS)
    genre_map = client.get_genre_map()
    catalog_dict: dict[int, dict] = {}
    pages_done = 0

    for sort_by in _SORT_ORDERS:
        for page in range(1, pages_per_order + 1):
            try:
                results = client.discover_movies(page=page, sort_by=sort_by)
            except Exception:
                break
            for movie in results:
                mid = movie["id"]
                if mid not in seen_ids and mid not in catalog_dict:
                    catalog_dict[mid] = {
                        "tmdb_id": mid,
                        "title": movie["title"],
                        "year": movie.get("release_date", "")[:4],
                        "poster_path": movie.get("poster_path"),
                        "overview": movie.get("overview") or "",
                        "tmdb_rating": movie.get("vote_average", 0),
                        "genres": [
                            genre_map.get(gid, "")
                            for gid in movie.get("genre_ids", [])
                        ],
                        "directors": [],
                        "cast": [],
                        "keywords": [],
                    }
            pages_done += 1
            if progress_callback:
                progress_callback(pages_done, total_pages)
            time.sleep(0.05)

    catalog = list(catalog_dict.values())

    if not catalog:
        return []

    # 4. Batch-encode and score in the weighted space
    catalog_embeddings = encode([m["overview"] for m in catalog])
    cat_norms = np.linalg.norm(catalog_embeddings, axis=1, keepdims=True)
    normed_catalog = catalog_embeddings / np.where(cat_norms > 0, cat_norms, 1.0)
    normed_catalog_w = _apply_weights_and_normalise(normed_catalog, dim_w)

    cos_sims = normed_catalog_w @ taste_vector                     # (N_catalog,)

    # 5. Fetch full details for top candidates and add categorical bonus
    top_k = min(n * 5, len(catalog))
    top_indices = np.argsort(cos_sims)[::-1][:top_k]

    final: list[dict] = []
    for idx in top_indices:
        movie = catalog[idx]
        try:
            details = client.get_movie_details(movie["tmdb_id"])
        except Exception:
            movie["_cos_sim"] = float(cos_sims[idx])
            final.append(movie)
            continue
        movie.update({
            "poster_path": details.get("poster_path"),
            **client.extract_features(details),
        })
        movie["_cos_sim"] = float(cos_sims[idx])
        final.append(movie)
        time.sleep(0.05)

    if not final:
        return []

    cat_scores = np.array([_categorical_bonus(m, profile) for m in final])
    cat_max = np.abs(cat_scores).max() or 1.0
    for movie, cat in zip(final, cat_scores):
        movie["score"] = movie["_cos_sim"] + 0.2 * float(cat / cat_max)
        movie["predicted_rating"] = round(
            float(np.clip(5.0 + 4.0 * movie["score"], 1.0, 10.0)), 1
        )

    return sorted(final, key=lambda m: m["score"], reverse=True)[:n]
