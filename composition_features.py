"""
Composition Feature Engineering
-------------------------------
Classify 5-champion compositions along strategic axes
and apply PCA to champion interaction matrices.
"""

from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
# Champion Classification Labels
# ============================================================================
# These are approximate timing/role labels for common champions.
# In production, derive these from per-patch win-rate-by-game-length curves.

CHAMPION_TIMING = {
    # Early game champions (power spike < 25 min)
    "Renekton": "early", "Lee Sin": "early", "Elise": "early",
    "Pantheon": "early", "Jayce": "early", "Draven": "early",
    "Lucian": "early", "Nidalee": "early", "Rek'Sai": "early",
    "Rumble": "early", "Olaf": "early", "LeBlanc": "early",
    "Syndra": "early", "Graves": "early",
    # Mid game champions
    "Ahri": "mid", "Viktor": "mid", "Orianna": "mid",
    "Gnar": "mid", "Jarvan IV": "mid", "Xin Zhao": "mid",
    "Ezreal": "mid", "Jhin": "mid", "Ashe": "mid",
    "Viego": "mid", "Akali": "mid", "Sylas": "mid",
    "Twisted Fate": "mid", "Corki": "mid", "Zeri": "mid",
    "Nautilus": "mid", "Thresh": "mid", "Rakan": "mid",
    "Alistar": "mid", "Leona": "mid",
    # Late game champions (power spike > 35 min)
    "Kayle": "late", "Kassadin": "late", "Vayne": "late",
    "Kog'Maw": "late", "Jinx": "late", "Azir": "late",
    "Ryze": "late", "Gangplank": "late", "Vladimir": "late",
    "Aphelios": "late", "Kai'Sa": "late", "Twitch": "late",
    "Senna": "late", "Smolder": "late", "Sivir": "late",
    "Ornn": "late", "Sion": "late",
}

CHAMPION_ROLE_STYLE = {
    # Lane-dominant top/jg
    "Renekton": "lane_dominant", "Jayce": "lane_dominant",
    "Darius": "lane_dominant", "Aatrox": "lane_dominant",
    "Rumble": "lane_dominant", "Kennen": "lane_dominant",
    # Skirmish-focused top/jg
    "Lee Sin": "skirmish", "Viego": "skirmish", "Nidalee": "skirmish",
    "Graves": "skirmish", "Kindred": "skirmish",
    "Fiora": "skirmish", "Irelia": "skirmish", "Camille": "skirmish",
    "Jax": "skirmish", "Yone": "skirmish",
    # Teamfight-focused
    "Ornn": "teamfight", "Sejuani": "teamfight", "Maokai": "teamfight",
    "Jarvan IV": "teamfight", "Wukong": "teamfight",
    "Amumu": "teamfight", "Malphite": "teamfight", "Sion": "teamfight",
}

BOTLANE_CARRY_TYPE = {
    # Scaling ADCs
    "Jinx": "scaling", "Aphelios": "scaling", "Kog'Maw": "scaling",
    "Vayne": "scaling", "Twitch": "scaling", "Kai'Sa": "scaling",
    "Zeri": "scaling", "Smolder": "scaling", "Sivir": "scaling",
    # Utility / early-focused bot
    "Ashe": "utility", "Jhin": "utility", "Varus": "utility",
    "Senna": "utility", "Ezreal": "utility", "Kalista": "utility",
    "Xayah": "utility",
    # Lane bully ADCs
    "Draven": "lane_bully", "Lucian": "lane_bully",
    "Caitlyn": "lane_bully", "Miss Fortune": "lane_bully",
    "Tristana": "lane_bully",
}

TEAMFIGHT_CHAMPIONS = {
    "Ornn", "Sejuani", "Maokai", "Amumu", "Malphite", "Wukong",
    "Jarvan IV", "Orianna", "Azir", "Kennen", "Rumble", "Sion",
    "Rakan", "Alistar", "Leona", "Nautilus",
    "Jinx", "Aphelios", "Kog'Maw", "Sivir", "Miss Fortune",
}

PICK_COMP_CHAMPIONS = {
    "Ahri", "LeBlanc", "Nidalee", "Thresh", "Blitzcrank",
    "Lee Sin", "Elise", "Camille", "Syndra", "Zoe",
    "Ashe", "Varus", "Jhin", "Twisted Fate", "Pyke",
    "Rengar", "Kha'Zix", "Evelynn",
}


class CompositionFeatures:
    """
    Classify champion compositions along strategic axes and extract
    PCA features from champion interaction matrices.
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = load_config(config_path)
        comp_cfg = cfg.get("composition", {})
        self.pca_components = comp_cfg.get("pca_components", 8)
        self.min_champion_appearances = comp_cfg.get("min_champion_appearances", 20)
        self.early_threshold = comp_cfg.get("early_game_threshold_minutes", 25.0)
        self.late_threshold = comp_cfg.get("late_game_threshold_minutes", 35.0)

        self.pca: Optional[PCA] = None
        self.interaction_champions: Optional[list] = None
        self.scaler = StandardScaler()

    # ========================================================================
    # Axis Classification
    # ========================================================================

    @staticmethod
    def classify_timing(champions: list) -> float:
        """
        Score a composition's timing: -1 = full early, +1 = full late.

        Args:
            champions: List of 5 champion names.

        Returns:
            Float in [-1, 1]. Negative = early-focused, Positive = late-focused.
        """
        timing_scores = {"early": -1.0, "mid": 0.0, "late": 1.0}
        scores = []
        for champ in champions:
            timing = CHAMPION_TIMING.get(champ, "mid")
            scores.append(timing_scores[timing])
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def classify_topjg_priority(champions: list, positions: list) -> float:
        """
        Score top-jungle priority: -1 = lane-dominant, +1 = skirmish-focused.

        Args:
            champions: List of 5 champion names.
            positions: List of 5 position labels (top, jng, mid, bot, sup).

        Returns:
            Float in [-1, 1].
        """
        style_scores = {"lane_dominant": -1.0, "skirmish": 1.0, "teamfight": 0.0}
        scores = []
        for champ, pos in zip(champions, positions):
            if pos in ("top", "jng"):
                style = CHAMPION_ROLE_STYLE.get(champ, "teamfight")
                scores.append(style_scores.get(style, 0.0))
        return float(np.mean(scores)) if scores else 0.0

    @staticmethod
    def classify_botlane_carry(champions: list, positions: list) -> float:
        """
        Score bot lane carry dependency: -1 = utility, +1 = scaling.

        Args:
            champions: List of 5 champion names.
            positions: List of 5 position labels.

        Returns:
            Float in [-1, 1].
        """
        type_scores = {"utility": -1.0, "lane_bully": 0.0, "scaling": 1.0}
        for champ, pos in zip(champions, positions):
            if pos == "bot":
                carry_type = BOTLANE_CARRY_TYPE.get(champ, "utility")
                return type_scores.get(carry_type, 0.0)
        return 0.0

    @staticmethod
    def classify_teamfight_vs_pick(champions: list) -> float:
        """
        Score composition style: -1 = pick comp, +1 = teamfight comp.

        Args:
            champions: List of 5 champion names.

        Returns:
            Float in [-1, 1].
        """
        tf_count = sum(1 for c in champions if c in TEAMFIGHT_CHAMPIONS)
        pick_count = sum(1 for c in champions if c in PICK_COMP_CHAMPIONS)
        total = tf_count + pick_count
        if total == 0:
            return 0.0
        return (tf_count - pick_count) / total

    def classify_composition(self, champions: list, positions: list) -> dict:
        """
        Classify a 5-champion composition along all 4 axes.

        Args:
            champions: List of 5 champion names.
            positions: List of 5 position labels.

        Returns:
            Dict with keys: timing, topjg_priority, botlane_carry, teamfight_vs_pick
        """
        return {
            "timing": self.classify_timing(champions),
            "topjg_priority": self.classify_topjg_priority(champions, positions),
            "botlane_carry": self.classify_botlane_carry(champions, positions),
            "teamfight_vs_pick": self.classify_teamfight_vs_pick(champions),
        }

    # ========================================================================
    # Champion Interaction Matrix + PCA
    # ========================================================================

    def build_interaction_matrix(self, draft_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a 10v10 champion co-occurrence / interaction matrix.

        For each game, create indicator vectors for both teams' champions,
        then record the co-occurrence. This captures which champions tend
        to appear together or against each other.

        Args:
            draft_df: DataFrame with columns [gameid, side, champion].

        Returns:
            DataFrame where each row is a game and columns are champion indicators.
        """
        if draft_df.empty:
            return pd.DataFrame()

        # Count champion appearances
        champ_counts = draft_df["champion"].value_counts()
        valid_champs = champ_counts[champ_counts >= self.min_champion_appearances].index.tolist()
        self.interaction_champions = sorted(valid_champs)

        if len(self.interaction_champions) < 2:
            return pd.DataFrame()

        # Pivot: each game-side gets an indicator row
        game_champs = draft_df[draft_df["champion"].isin(valid_champs)].copy()

        # Create side-specific indicators
        records = []
        for gameid, group in game_champs.groupby("gameid"):
            row = {"gameid": gameid}
            for _, pick in group.iterrows():
                side = pick.get("side", "Blue")
                champ = pick["champion"]
                prefix = "ally_" if side == "Blue" else "enemy_"
                row[f"{prefix}{champ}"] = 1
            records.append(row)

        interaction_df = pd.DataFrame(records).fillna(0)
        interaction_df = interaction_df.set_index("gameid")
        return interaction_df

    def fit_pca(self, draft_df: pd.DataFrame) -> np.ndarray:
        """
        Fit PCA on the champion interaction matrix.

        Args:
            draft_df: Draft DataFrame with [gameid, side, champion].

        Returns:
            Array of PCA-transformed features (n_games x n_components).
        """
        interaction = self.build_interaction_matrix(draft_df)
        if interaction.empty:
            return np.array([])

        # Standardize
        scaled = self.scaler.fit_transform(interaction.values)

        # Fit PCA
        n_components = min(self.pca_components, scaled.shape[1], scaled.shape[0])
        self.pca = PCA(n_components=n_components)
        transformed = self.pca.fit_transform(scaled)

        return transformed

    def transform_pca(self, draft_df: pd.DataFrame) -> np.ndarray:
        """
        Transform new draft data using the fitted PCA.

        Args:
            draft_df: Draft DataFrame.

        Returns:
            PCA-transformed features.
        """
        if self.pca is None:
            raise RuntimeError("PCA not fitted. Call fit_pca first.")

        interaction = self.build_interaction_matrix(draft_df)
        if interaction.empty:
            return np.array([])

        scaled = self.scaler.transform(interaction.values)
        return self.pca.transform(scaled)

    def compute_game_features(
        self,
        games_df: pd.DataFrame,
        draft_df: pd.DataFrame,
        player_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add composition features to a game-level DataFrame.

        For each game, classify both teams' compositions along 4 axes
        and compute the delta (team_a - team_b). Also adds PCA features.

        Args:
            games_df: Game-level DataFrame with team_a, team_b columns.
            draft_df: Draft picks DataFrame with gameid, side, champion, position.
            player_df: Player-level DataFrame for position assignments.

        Returns:
            games_df with added composition feature columns.
        """
        df = games_df.copy()

        # Initialize columns
        axis_names = ["timing", "topjg_priority", "botlane_carry", "teamfight_vs_pick"]
        for axis in axis_names:
            df[f"comp_{axis}_a"] = 0.0
            df[f"comp_{axis}_b"] = 0.0
            df[f"comp_{axis}_delta"] = 0.0

        # Classify compositions per game
        for idx, row in df.iterrows():
            gameid = row.get("gameid")
            if gameid is None:
                continue

            game_draft = draft_df[draft_df["gameid"] == gameid] if not draft_df.empty else pd.DataFrame()
            if game_draft.empty:
                continue

            for side, team_key in [("Blue", "a"), ("Red", "b")]:
                side_picks = game_draft[game_draft["side"] == side]
                if side_picks.empty:
                    continue

                champs = side_picks["champion"].tolist()
                positions = side_picks["position"].tolist() if "position" in side_picks.columns else ["top", "jng", "mid", "bot", "sup"]

                classification = self.classify_composition(champs, positions)
                for axis in axis_names:
                    df.at[idx, f"comp_{axis}_{team_key}"] = classification[axis]

            # Compute deltas (team_a - team_b)
            for axis in axis_names:
                df.at[idx, f"comp_{axis}_delta"] = (
                    df.at[idx, f"comp_{axis}_a"] - df.at[idx, f"comp_{axis}_b"]
                )

        # PCA features from interaction matrix
        if not draft_df.empty:
            pca_result = self.fit_pca(draft_df)
            if pca_result.size > 0:
                game_ids_in_pca = draft_df.groupby("gameid").first().index
                pca_df = pd.DataFrame(
                    pca_result,
                    index=game_ids_in_pca[:len(pca_result)],
                    columns=[f"comp_pca_{i}" for i in range(pca_result.shape[1])],
                )
                df = df.merge(pca_df, left_on="gameid", right_index=True, how="left")

        # Fill missing PCA columns with 0
        for i in range(self.pca_components):
            col = f"comp_pca_{i}"
            if col not in df.columns:
                df[col] = 0.0
            else:
                df[col] = df[col].fillna(0.0)

        return df


if __name__ == "__main__":
    print("=" * 50)
    print("Composition Features — Demo")
    print("=" * 50)

    cf = CompositionFeatures.__new__(CompositionFeatures)
    cf.pca_components = 8
    cf.min_champion_appearances = 2
    cf.early_threshold = 25.0
    cf.late_threshold = 35.0
    cf.pca = None
    cf.interaction_champions = None
    cf.scaler = StandardScaler()

    # Example composition
    champs_a = ["Jayce", "Lee Sin", "Ahri", "Jinx", "Thresh"]
    pos = ["top", "jng", "mid", "bot", "sup"]
    result = cf.classify_composition(champs_a, pos)
    print(f"\nTeam A: {champs_a}")
    for k, v in result.items():
        print(f"  {k}: {v:.2f}")

    champs_b = ["Ornn", "Sejuani", "Azir", "Aphelios", "Rakan"]
    result_b = cf.classify_composition(champs_b, pos)
    print(f"\nTeam B: {champs_b}")
    for k, v in result_b.items():
        print(f"  {k}: {v:.2f}")

    print("\nDeltas (A - B):")
    for k in result:
        print(f"  {k}: {result[k] - result_b[k]:.2f}")
