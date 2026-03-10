"""
Archetype Model
---------------
Compute team performance vectors, pairwise cosine similarity,
and head-to-head correction factors based on stylistic matchups.
"""

from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class ArchetypeModel:
    """
    Builds team archetype vectors from aggregate performance stats
    and computes pairwise similarity for head-to-head corrections.

    Each team is represented by a vector:
        [avg_gold_diff_15, first_dragon_pct, first_herald_pct,
         kd_ratio, avg_game_length]
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = load_config(config_path)
        arch_cfg = cfg.get("archetype", {})
        self.feature_names = arch_cfg.get("features", [
            "avg_gold_diff_15", "first_dragon_pct", "first_herald_pct",
            "kd_ratio", "avg_game_length",
        ])
        self.reference_team = arch_cfg.get("reference_team", "Gen.G")
        self.min_games = arch_cfg.get("min_games", 10)
        self.lookback_games = arch_cfg.get("lookback_games", 0)

        self.team_vectors: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[pd.DataFrame] = None
        self.scaler = StandardScaler()

    def compute_team_stats(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute aggregate performance stats per team from team-level rows.

        Expects Oracle's Elixir team-level rows with columns:
        teamname, golddiffat15, firstdragon, firstherald, kills, deaths, gamelength

        Returns:
            DataFrame indexed by teamname with archetype feature columns.
        """
        if team_df.empty:
            return pd.DataFrame()

        df = team_df.copy()

        # Apply lookback filter if configured
        if self.lookback_games > 0 and "date" in df.columns:
            df = df.sort_values("date")
            df = df.groupby("teamname").tail(self.lookback_games)

        # Compute per-team aggregates
        agg = df.groupby("teamname").agg(
            avg_gold_diff_15=("golddiffat15", "mean"),
            first_dragon_pct=("firstdragon", "mean"),
            first_herald_pct=("firstherald", "mean"),
            total_kills=("kills", "sum"),
            total_deaths=("deaths", "sum"),
            avg_game_length=("gamelength", "mean"),
            num_games=("teamname", "size"),
        )

        # KD ratio (avoid division by zero)
        agg["kd_ratio"] = agg["total_kills"] / agg["total_deaths"].replace(0, 1)

        # Filter teams with insufficient games
        agg = agg[agg["num_games"] >= self.min_games]

        # Keep only archetype features
        available_features = [f for f in self.feature_names if f in agg.columns]
        return agg[available_features + ["num_games"]]

    def build_vectors(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build normalized team archetype vectors.

        Args:
            team_df: Team-level DataFrame from Oracle's Elixir data.

        Returns:
            DataFrame of normalized team vectors (z-scored).
        """
        stats = self.compute_team_stats(team_df)
        if stats.empty:
            self.team_vectors = pd.DataFrame()
            return self.team_vectors

        feature_cols = [c for c in self.feature_names if c in stats.columns]
        raw = stats[feature_cols].copy()

        # Fill NaN with column medians before scaling
        raw = raw.fillna(raw.median())

        # Z-score normalize
        normalized = pd.DataFrame(
            self.scaler.fit_transform(raw),
            index=raw.index,
            columns=raw.columns,
        )
        normalized["num_games"] = stats["num_games"]
        self.team_vectors = normalized
        return self.team_vectors

    def compute_similarity_matrix(self) -> pd.DataFrame:
        """
        Compute pairwise cosine similarity between all team vectors.

        Returns:
            Square DataFrame of cosine similarities.
        """
        if self.team_vectors is None or self.team_vectors.empty:
            return pd.DataFrame()

        feature_cols = [c for c in self.feature_names if c in self.team_vectors.columns]
        vectors = self.team_vectors[feature_cols].values

        sim = cosine_similarity(vectors)
        teams = self.team_vectors.index.tolist()
        self.similarity_matrix = pd.DataFrame(sim, index=teams, columns=teams)
        return self.similarity_matrix

    def get_similarity(self, team_a: str, team_b: str) -> float:
        """
        Get cosine similarity between two teams.

        Returns:
            Float in [-1, 1]. Higher = more stylistically similar.
            Returns 0.0 if either team is unknown.
        """
        if self.similarity_matrix is None or self.similarity_matrix.empty:
            return 0.0
        if team_a not in self.similarity_matrix.index or team_b not in self.similarity_matrix.columns:
            return 0.0
        return float(self.similarity_matrix.loc[team_a, team_b])

    def teams_closest_to_reference(self, top_n: int = 10) -> list:
        """
        Find teams most stylistically similar to the reference team.

        Returns:
            List of (team_name, similarity_score) tuples, sorted descending.
        """
        if self.similarity_matrix is None or self.reference_team not in self.similarity_matrix.index:
            return []

        sims = self.similarity_matrix[self.reference_team].drop(self.reference_team, errors="ignore")
        ranked = sims.sort_values(ascending=False)
        return list(ranked.head(top_n).items())

    def head_to_head_correction(self, team_a: str, team_b: str) -> float:
        """
        Compute a head-to-head correction factor based on archetype similarity.

        Logic: when two teams have very similar styles, upsets are more likely
        (the matchup is more volatile). When styles differ significantly,
        the stronger team's structural advantages are more pronounced.

        Returns:
            Float correction factor:
            - Close to 0 when teams are very similar (no strong edge from style)
            - Positive when team_a has a stylistic edge over team_b
            - Negative when team_b has a stylistic edge
        """
        if self.team_vectors is None or self.team_vectors.empty:
            return 0.0

        feature_cols = [c for c in self.feature_names if c in self.team_vectors.columns]

        if team_a not in self.team_vectors.index or team_b not in self.team_vectors.index:
            return 0.0

        vec_a = self.team_vectors.loc[team_a, feature_cols].values
        vec_b = self.team_vectors.loc[team_b, feature_cols].values

        # Signed difference in archetype vectors
        diff = vec_a - vec_b

        # Weighted sum: positive features (gold, objectives) favor team_a
        # Use equal weights for simplicity; tunable in config
        correction = float(np.mean(diff))

        # Scale by dissimilarity — bigger correction when teams play differently
        sim = self.get_similarity(team_a, team_b)
        dissimilarity_scale = 1.0 - max(0, sim)  # [0, 2] range

        return correction * dissimilarity_scale

    def compute_game_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add archetype features to a game-level DataFrame.

        Adds:
        - archetype_similarity: cosine similarity between the two teams
        - archetype_similarity_delta: head-to-head correction factor

        Args:
            games_df: Game-level DataFrame with team_a, team_b columns.

        Returns:
            DataFrame with archetype columns added.
        """
        df = games_df.copy()
        df["archetype_similarity"] = df.apply(
            lambda r: self.get_similarity(r["team_a"], r["team_b"]), axis=1
        )
        df["archetype_similarity_delta"] = df.apply(
            lambda r: self.head_to_head_correction(r["team_a"], r["team_b"]), axis=1
        )
        return df


if __name__ == "__main__":
    print("=" * 50)
    print("Archetype Model — Demo")
    print("=" * 50)

    # Synthetic team data
    np.random.seed(42)
    teams = ["Gen.G", "T1", "HLE", "KT", "DRX", "BRO"]
    rows = []
    for t in teams:
        for _ in range(20):
            rows.append({
                "teamname": t,
                "golddiffat15": np.random.normal(500 if t in ["Gen.G", "T1"] else -200, 1500),
                "firstdragon": np.random.binomial(1, 0.6 if t == "Gen.G" else 0.45),
                "firstherald": np.random.binomial(1, 0.55 if t in ["Gen.G", "T1"] else 0.4),
                "kills": np.random.randint(5, 25),
                "deaths": np.random.randint(5, 25),
                "gamelength": np.random.normal(1800, 300),
            })
    team_df = pd.DataFrame(rows)

    model = ArchetypeModel.__new__(ArchetypeModel)
    model.feature_names = ["avg_gold_diff_15", "first_dragon_pct", "first_herald_pct", "kd_ratio", "avg_game_length"]
    model.reference_team = "Gen.G"
    model.min_games = 10
    model.lookback_games = 0
    model.scaler = StandardScaler()
    model.team_vectors = None
    model.similarity_matrix = None

    vectors = model.build_vectors(team_df)
    print("\nTeam Archetype Vectors (normalized):")
    print(vectors.round(2))

    sim = model.compute_similarity_matrix()
    print("\nPairwise Similarity Matrix:")
    print(sim.round(2))

    print(f"\nTeams most similar to {model.reference_team}:")
    for team, score in model.teams_closest_to_reference(5):
        print(f"  {team:<12} {score:.3f}")
