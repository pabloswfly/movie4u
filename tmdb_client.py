from typing import Optional

import requests

_TMDB_BASE = "https://api.themoviedb.org/3"
_POSTER_BASE = "https://image.tmdb.org/t/p/w342"


class TMDBClient:
    def __init__(self, api_key: str):
        self._session = requests.Session()
        self._session.params = {"api_key": api_key}  # type: ignore[assignment]

    def _get(self, path: str, **params) -> dict:
        resp = self._session.get(
            f"{_TMDB_BASE}{path}", params=params, timeout=10
        )
        resp.raise_for_status()
        return resp.json()

    def search_movie(
        self, title: str, year: Optional[int] = None
    ) -> Optional[dict]:
        """Return the first TMDB search result for a title (+ optional year)."""
        params: dict = {"query": title, "include_adult": False}
        if year:
            params["year"] = year
        results = self._get("/search/movie", **params).get("results", [])
        return results[0] if results else None

    def get_movie_details(self, tmdb_id: int) -> dict:
        """Return full movie details including credits and keywords."""
        return self._get(
            f"/movie/{tmdb_id}",
            append_to_response="credits,keywords",
        )

    def get_genre_map(self) -> dict[int, str]:
        """Return a mapping of TMDB genre ID → genre name."""
        genres = self._get("/genre/movie/list").get("genres", [])
        return {g["id"]: g["name"] for g in genres}

    def discover_movies(
        self,
        page: int = 1,
        min_votes: int = 200,
        sort_by: str = "vote_count.desc",
    ) -> list[dict]:
        """Fetch one page of movies from the TMDB discover endpoint."""
        return self._get(
            "/discover/movie",
            sort_by=sort_by,
            vote_count_gte=min_votes,
            page=page,
        ).get("results", [])

    @staticmethod
    def poster_url(path: Optional[str]) -> Optional[str]:
        return f"{_POSTER_BASE}{path}" if path else None

    @staticmethod
    def extract_features(details: dict) -> dict:
        """Pull genres, directors, cast, keywords out of a details response."""
        credits = details.get("credits", {})
        return {
            "genres": [g["name"] for g in details.get("genres", [])],
            "directors": [
                c["name"]
                for c in credits.get("crew", [])
                if c.get("job") == "Director"
            ],
            "cast": [c["name"] for c in credits.get("cast", [])[:5]],
            "keywords": [
                k["name"]
                for k in details.get("keywords", {}).get("keywords", [])
            ],
        }
