"""
Player Mastery Model
--------------------
Compute per-player champion mastery with recency weighting and
patch validity, then aggregate to team-level champion pool strength.
"""

from typing import Optional

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class PlayerMasteryModel:
    """
    Computes recency-weighted champion mastery per player
    and aggregates to team-level strength for a given draft.
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = load_config(config_path)
        pm_cfg = cfg.get("player_mastery", {})
        self.half_life_days = pm_cfg.get("recency_half_life_days", 30)
        self.min_games = pm_cfg.get("min_games", 3)
        self.patch_wr_threshold = pm_cfg.get("patch_winrate_change_threshold", 0.10)

        # Decay constant: lambda = ln(2) / half_life
        self.decay_lambda = np.log(2) / self.half_life_days

        self.player_mastery: Optional[pd.DataFrame] = None
        self.champion_patch_stats: Optional[pd.DataFrame] = None

    def compute_player_champion_stats(
        self,
        player_df: pd.DataFrame,
        reference_date: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """
        Compute per-player, per-champion mastery statistics.

        For each (player, champion) pair:
        - games: total competitive games
        - wins: total wins
        - win_rate: win / games
        - recency_score: exponential-decay weighted score

        Args:
            player_df: Player-level rows with playername, champion, result, date.
            reference_date: Date from which to compute recency decay.
                            Defaults to the latest date in the data.

        Returns:
            DataFrame with columns:
            [playername, champion, games, wins, win_rate, recency_score]
        """
        if player_df.empty:
            return pd.DataFrame()

        df = player_df.copy()

        # Ensure date is datetime
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")

        if reference_date is None and "date" in df.columns:
            reference_date = df["date"].max()

        # Compute recency weight per game
        if reference_date is not None and "date" in df.columns:
            df["days_ago"] = (reference_date - df["date"]).dt.total_seconds() / 86400
            df["days_ago"] = df["days_ago"].clip(lower=0)
            df["recency_weight"] = np.exp(-self.decay_lambda * df["days_ago"])
        else:
            df["recency_weight"] = 1.0

        # Ensure result is numeric
        if "result" in df.columns:
            df["result"] = pd.to_numeric(df["result"], errors="coerce").fillna(0)

        # Group by player + champion
        grouped = df.groupby(["playername", "champion"]).agg(
            games=("result", "size"),
            wins=("result", "sum"),
            recency_score=("recency_weight", lambda x: (x * df.loc[x.index, "result"]).sum() / x.sum()
                           if x.sum() > 0 else 0.0),
        ).reset_index()

        grouped["win_rate"] = grouped["wins"] / grouped["games"].replace(0, 1)

        # Filter by minimum games
        grouped = grouped[grouped["games"] >= self.min_games].copy()

        self.player_mastery = grouped
        return grouped

    def compute_champion_patch_stats(self, player_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-champion win rates by patch for patch validity checking.

        Returns:
            DataFrame with columns: [champion, patch, games, win_rate]
        """
        if player_df.empty or "patch" not in player_df.columns:
            return pd.DataFrame()

        df = player_df.copy()
        df["result"] = pd.to_numeric(df["result"], errors="coerce").fillna(0)

        patch_stats = df.groupby(["champion", "patch"]).agg(
            games=("result", "size"),
            wins=("result", "sum"),
        ).reset_index()

        patch_stats["win_rate"] = patch_stats["wins"] / patch_stats["games"].replace(0, 1)

        self.champion_patch_stats = patch_stats
        return patch_stats

    def get_patch_validity(self, champion: str, current_patch: str) -> bool:
        """
        Check if a champion's mastery should count on the current patch.

        A champion is invalid if its win rate changed by more than the
        threshold compared to the prior patch.

        Returns:
            True if mastery is valid, False if it should be zeroed out.
        """
        if self.champion_patch_stats is None or self.champion_patch_stats.empty:
            return True

        champ_data = self.champion_patch_stats[
            self.champion_patch_stats["champion"] == champion
        ].sort_values("patch")

        if len(champ_data) < 2:
            return True

        # Find current and prior patch stats
        patches = champ_data["patch"].unique()
        if current_patch not in patches:
            return True

        patch_idx = list(patches).index(current_patch)
        if patch_idx == 0:
            return True

        current_wr = champ_data[champ_data["patch"] == current_patch]["win_rate"].values[0]
        prior_patch = patches[patch_idx - 1]
        prior_wr = champ_data[champ_data["patch"] == prior_patch]["win_rate"].values[0]

        return abs(current_wr - prior_wr) <= self.patch_wr_threshold

    def get_player_mastery_score(
        self,
        player: str,
        champion: str,
        current_patch: Optional[str] = None,
    ) -> float:
        """
        Get the mastery score for a player on a specific champion.

        Returns:
            Recency-weighted mastery score, or 0.0 if below threshold
            or patch-invalidated.
        """
        if self.player_mastery is None or self.player_mastery.empty:
            return 0.0

        match = self.player_mastery[
            (self.player_mastery["playername"] == player) &
            (self.player_mastery["champion"] == champion)
        ]

        if match.empty:
            return 0.0

        score = float(match.iloc[0]["recency_score"])

        # Apply patch validity check
        if current_patch and not self.get_patch_validity(champion, current_patch):
            return 0.0

        return score

    def compute_team_mastery(
        self,
        players: list,
        champions: list,
        current_patch: Optional[str] = None,
    ) -> float:
        """
        Compute team-level champion pool strength for a drafted composition.

        Aggregates individual player mastery scores for their drafted champions
        into a single team score.

        Args:
            players: List of 5 player names.
            champions: List of 5 champions (matching player order).
            current_patch: Current patch for validity checking.

        Returns:
            Mean mastery score across the 5 players on their drafted champions.
        """
        scores = []
        for player, champ in zip(players, champions):
            score = self.get_player_mastery_score(player, champ, current_patch)
            scores.append(score)
        return float(np.mean(scores)) if scores else 0.0

    def compute_game_features(
        self,
        games_df: pd.DataFrame,
        draft_df: pd.DataFrame,
        current_patch: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Add player mastery features to a game-level DataFrame.

        For each game, compute team-level mastery for both sides
        and the delta (team_a - team_b).

        Args:
            games_df: Game-level DataFrame.
            draft_df: Draft picks with gameid, side, playername, champion.

        Returns:
            DataFrame with added mastery columns.
        """
        df = games_df.copy()
        df["mastery_a"] = 0.0
        df["mastery_b"] = 0.0
        df["player_mastery_delta"] = 0.0

        for idx, row in df.iterrows():
            gameid = row.get("gameid")
            patch = current_patch or row.get("patch")

            if gameid is None or draft_df.empty:
                continue

            game_draft = draft_df[draft_df["gameid"] == gameid]
            if game_draft.empty:
                continue

            for side, key in [("Blue", "a"), ("Red", "b")]:
                side_picks = game_draft[game_draft["side"] == side]
                if side_picks.empty or "playername" not in side_picks.columns:
                    continue

                players = side_picks["playername"].tolist()
                champs = side_picks["champion"].tolist()
                mastery = self.compute_team_mastery(players, champs, patch)
                df.at[idx, f"mastery_{key}"] = mastery

            df.at[idx, "player_mastery_delta"] = (
                df.at[idx, "mastery_a"] - df.at[idx, "mastery_b"]
            )

        return df


if __name__ == "__main__":
    print("=" * 50)
    print("Player Mastery Model — Demo")
    print("=" * 50)

    # Synthetic player data
    np.random.seed(42)
    records = []
    players = {
        "Zeus": ["Jayce", "Aatrox", "Kennen", "Gnar"],
        "Oner": ["Lee Sin", "Viego", "Graves", "Nidalee"],
        "Faker": ["Azir", "Ahri", "Orianna", "LeBlanc", "Akali"],
        "Gumayusi": ["Jinx", "Aphelios", "Ezreal", "Kai'Sa"],
        "Keria": ["Thresh", "Nautilus", "Rakan", "Alistar"],
    }

    base_date = pd.Timestamp("2024-06-01")
    for player, champ_pool in players.items():
        for champ in champ_pool:
            n_games = np.random.randint(5, 20)
            for g in range(n_games):
                records.append({
                    "playername": player,
                    "champion": champ,
                    "result": np.random.binomial(1, 0.6),
                    "date": base_date - pd.Timedelta(days=np.random.randint(1, 120)),
                    "patch": "14.10",
                })

    player_df = pd.DataFrame(records)

    model = PlayerMasteryModel.__new__(PlayerMasteryModel)
    model.half_life_days = 30
    model.min_games = 3
    model.patch_wr_threshold = 0.10
    model.decay_lambda = np.log(2) / 30
    model.player_mastery = None
    model.champion_patch_stats = None

    stats = model.compute_player_champion_stats(player_df, reference_date=base_date)
    print("\nTop mastery scores:")
    print(stats.sort_values("recency_score", ascending=False).head(10).to_string(index=False))

    # Compute team mastery for a draft
    draft_players = ["Zeus", "Oner", "Faker", "Gumayusi", "Keria"]
    draft_champs = ["Jayce", "Lee Sin", "Azir", "Jinx", "Thresh"]
    team_score = model.compute_team_mastery(draft_players, draft_champs)
    print(f"\nTeam mastery for T1 draft {draft_champs}: {team_score:.3f}")
