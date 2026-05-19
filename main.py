import os
import re
from collections import Counter
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from features import load_encoder
from recommender import enrich_with_tmdb, get_recommendations
from tmdb_client import TMDBClient
from viz import build_scatter_figure

load_dotenv()

_EXAMPLE = """\
The Godfather (1972): 10
Blade Runner (1982): 9
Mulholland Drive (2001): 8
The Big Lebowski: 7
Interstellar: 6
"""

st.set_page_config(
    page_title="Movie Recommender", page_icon="🎬", layout="wide"
)
st.title("🎬 Movie Recommender")
st.caption(
    "Enter films you've seen, get personalised recommendations. "
    "Ratings are optional (default 7/10 if omitted)."
)


def parse_movie_list(text: str) -> list[dict]:
    """Parse a free-text movie list into rating dicts.

    Accepted formats per line (mixed is fine):
      Title (Year): Rating   →  The Godfather (1972): 10
      Title (Year) - Rating  →  Blade Runner (1982) - 9
      Title: Rating          →  Inception: 8
      Title                  →  The Dark Knight
    """
    movies: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue

        rating = 10  # default when not specified
        m = re.search(r"[\-:]\s*(\d{1,2})\s*$", line)
        if m:
            try:
                rating = max(1, min(10, int(m.group(1))))
            except ValueError:
                pass
            line = line[: m.start()].strip()

        year: Optional[int] = None
        m = re.search(r"\((\d{4})\)", line)
        if m:
            year = int(m.group(1))
            line = (line[: m.start()] + line[m.end() :]).strip()

        title = line.strip(" ,-:")
        if title:
            movies.append(
                {"title": title, "year": year, "user_rating": rating}
            )

    return movies


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")
    tmdb_key = st.text_input(
        "TMDB API Key",
        value=os.getenv("TMDB_API_KEY", ""),
        type="password",
        help="Free key from themoviedb.org",
    )
    n_recs = st.slider("Recommendations to show", 5, 40, 20, 5)
    catalog_size = st.slider(
        "Catalog size (movies to scan)",
        100, 20_000, 1_000, 500,
        help="More movies = better recommendations but slower. "
             "Scans across vote count and popularity rankings.",
    )

# ── Movie list input ───────────────────────────────────────────────────────────
st.subheader("Your Movie List")
st.caption(
    "One film per line. Optionally add `(year)` and/or a rating out of 10 "
    "separated by `:` or `-`.  Example:"
)
st.code(_EXAMPLE.strip(), language=None)

movie_text = st.text_area(
    "Movies",
    height=250,
    placeholder=_EXAMPLE,
    label_visibility="collapsed",
)

run = st.button("Get Recommendations", type="primary")

# ── Main ───────────────────────────────────────────────────────────────────────
if run:
    if not tmdb_key:
        st.error("Enter your TMDB API key in the sidebar.")
        st.stop()

    movies = parse_movie_list(movie_text)
    if not movies:
        st.error("No movies found — paste at least one title above.")
        st.stop()

    st.success(f"Parsed **{len(movies)}** movies from your list.")

    # 1. Enrich with TMDB
    enrich_bar = st.progress(0, text="Matching movies on TMDB…")

    def _enrich_progress(done, total):
        enrich_bar.progress(
            done / total, text=f"Matching movies on TMDB… {done}/{total}"
        )

    client = TMDBClient(tmdb_key)
    enriched, not_found = enrich_with_tmdb(
        movies, client, progress_callback=_enrich_progress
    )
    enrich_bar.empty()

    if not enriched:
        st.error(
            "Could not match any movies on TMDB. Check titles or your API key."
        )
        st.stop()

    msg = f"Matched **{len(enriched)}/{len(movies)}** movies on TMDB."
    if not_found:
        msg += f" **{len(not_found)} not found.**"
    st.info(msg)

    if not_found:
        with st.expander(f"Titles not found on TMDB ({len(not_found)})"):
            for title in not_found:
                st.write(f"- {title}")
            st.caption("Tip: add the release year to improve matching, e.g. `Dune (2021)`.")

    # 2. Taste profile summary
    st.subheader("Your Taste Profile")
    c1, c2, c3 = st.columns(3)

    genre_counts: Counter = Counter()
    director_counts: Counter = Counter()
    high_rated = [m for m in enriched if m["user_rating"] >= 7]

    for m in high_rated:
        genre_counts.update(m.get("genres", []))
        director_counts.update(m.get("directors", []))

    avg_rating = sum(m["user_rating"] for m in enriched) / len(enriched)

    with c1:
        st.metric("Avg rating you gave", f"{avg_rating:.1f} / 10")
        st.metric("Films rated ≥ 7", len(high_rated))

    with c2:
        st.markdown("**Top genres (liked films)**")
        for genre, cnt in genre_counts.most_common(6):
            st.write(f"- {genre} ({cnt})")

    with c3:
        st.markdown("**Favourite directors (liked films)**")
        for director, cnt in director_counts.most_common(6):
            st.write(f"- {director} ({cnt})")

    # 3. Recommendations
    with st.spinner("Loading sentence-transformer model…"):
        load_encoder()

    rec_bar = st.progress(0, text="Scanning TMDB catalog…")

    def _rec_progress(done, total):
        rec_bar.progress(
            done / total,
            text=f"Scanning catalog… {done * 20:,}/{total * 20:,} movies",
        )

    recs = get_recommendations(
        enriched, client, n=n_recs,
        catalog_size=catalog_size,
        progress_callback=_rec_progress,
    )
    rec_bar.empty()

    st.subheader(f"Top {len(recs)} Recommendations for You")

    cols_per_row = 4
    for row_start in range(0, len(recs), cols_per_row):
        cols = st.columns(cols_per_row)
        for col, movie in zip(
            cols, recs[row_start : row_start + cols_per_row]
        ):
            with col:
                poster = TMDBClient.poster_url(movie.get("poster_path"))
                if poster:
                    st.image(poster, use_container_width=True)
                else:
                    st.markdown("*(no poster)*")

                year = movie.get("year", "")
                st.markdown(
                    f"**{movie['title']}** {f'({year})' if year else ''}"
                )
                st.caption(
                    f"⭐ {movie['tmdb_rating']:.1f}  |  Predicted: {movie['predicted_rating']:.1f}/10"
                )

                genres = ", ".join(movie.get("genres", [])[:3])
                if genres:
                    st.caption(genres)

                directors = ", ".join(movie.get("directors", []))
                if directors:
                    st.caption(f"Dir. {directors}")

                with st.expander("Synopsis"):
                    st.write(movie.get("overview") or "No synopsis available.")

    # 4. Feature-space visualisation
    st.subheader("Explore the Feature Space")
    st.caption(
        "2D projection of synopsis embeddings. "
        "Circles = your movies (red → green = low → high rating). "
        "Stars = recommendations (darker blue = better match). "
        "Hover for details, scroll to zoom, drag to pan."
    )
    with st.spinner("Computing 2D projection…"):
        fig = build_scatter_figure(enriched, recs)
    st.plotly_chart(fig, use_container_width=True)
