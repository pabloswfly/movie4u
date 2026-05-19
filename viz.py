from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from features import encode


def _reduce_2d(embeddings: np.ndarray) -> np.ndarray:
    n = len(embeddings)
    if n > 10:
        try:
            import umap  # type: ignore[import]
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(15, n - 1),
                random_state=42,
            )
            return reducer.fit_transform(embeddings)
        except Exception:
            pass
    from sklearn.decomposition import PCA

    n_comp = min(2, n)
    coords = PCA(
        n_components=n_comp, random_state=42
    ).fit_transform(embeddings)
    if n_comp == 1:
        coords = np.hstack([coords, np.zeros_like(coords)])
    return coords


def build_scatter_figure(enriched: list[dict], recs: list[dict]) -> go.Figure:
    """Interactive 2-D embedding scatter of rated movies + recommendations."""
    all_movies = enriched + recs
    embeddings = encode([m.get("overview") or "" for m in all_movies])
    coords = _reduce_2d(embeddings)

    n_rated = len(enriched)
    rc = coords[:n_rated]
    ec = coords[n_rated:]

    fig = go.Figure()

    # Rated movies — labelled circles coloured by user rating
    fig.add_trace(go.Scatter(
        x=rc[:, 0],
        y=rc[:, 1],
        mode="markers+text",
        name="Your movies",
        text=[m["title"] for m in enriched],
        textposition="top center",
        textfont=dict(size=10),
        marker=dict(
            size=16,
            color=[m["user_rating"] for m in enriched],
            colorscale="RdYlGn",
            cmin=1,
            cmax=10,
            showscale=True,
            colorbar=dict(title="Your rating", x=1.02, len=0.55, y=0.72),
            line=dict(width=1.5, color="black"),
            symbol="circle",
        ),
        customdata=[
            [
                m["title"],
                m.get("year", ""),
                m["user_rating"],
                ", ".join(m.get("genres", [])[:3]),
                (m.get("overview") or "")[:140],
            ]
            for m in enriched
        ],
        hovertemplate=(
            "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
            "Your rating: <b>%{customdata[2]}/10</b><br>"
            "Genres: %{customdata[3]}<br>"
            "<i>%{customdata[4]}…</i>"
            "<extra></extra>"
        ),
    ))

    # Recommendations — stars coloured by match score
    if recs:
        scores = [m["score"] for m in recs]
        s_min, s_max = min(scores), max(scores)
        if s_min == s_max:
            s_max = s_min + 1e-6
        fig.add_trace(go.Scatter(
            x=ec[:, 0],
            y=ec[:, 1],
            mode="markers",
            name="Recommendations",
            marker=dict(
                size=11,
                color=scores,
                colorscale="Blues",
                cmin=s_min,
                cmax=s_max,
                showscale=True,
                colorbar=dict(
                    title="Match score", x=1.13, len=0.55, y=0.2
                ),
                line=dict(width=1, color="steelblue"),
                symbol="star",
            ),
            customdata=[
                [
                    m["title"],
                    m.get("year", ""),
                    f"{m['tmdb_rating']:.1f}",
                    f"{m['predicted_rating']:.1f}",
                    ", ".join(m.get("genres", [])[:3]),
                    (m.get("overview") or "")[:140],
                ]
                for m in recs
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                "TMDB: %{customdata[2]} ⭐  |  "
                "Predicted: <b>%{customdata[3]}/10</b><br>"
                "Genres: %{customdata[4]}<br>"
                "<i>%{customdata[5]}…</i>"
                "<extra></extra>"
            ),
        ))

    fig.update_layout(
        title="Movie Embedding Space — 2D Projection",
        xaxis=dict(title="Component 1", showgrid=False, zeroline=False),
        yaxis=dict(title="Component 2", showgrid=False, zeroline=False),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1.0,
            xanchor="left",
            x=0.0,
        ),
        height=660,
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_size=12, namelength=0),
        plot_bgcolor="rgba(235,235,235,0.5)",
        paper_bgcolor="white",
    )

    return fig
