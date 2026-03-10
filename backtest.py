"""
Backtest
--------
Simulate betting when |model_prob - market_prob| > edge threshold.
Supports flat betting and Kelly criterion sizing.
"""

from typing import Optional

import numpy as np
import pandas as pd
import yaml

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class Backtest:
    """
    Simulate betting on games where the model finds an edge over market odds.

    Strategies:
    - flat: bet a fixed fraction of bankroll on each qualifying game
    - kelly: bet using fractional Kelly criterion for optimal growth
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = load_config(config_path)
        bt_cfg = cfg.get("backtest", {})
        self.edge_threshold = bt_cfg.get("edge_threshold", 0.05)
        self.starting_bankroll = bt_cfg.get("starting_bankroll", 1000.0)
        self.bet_fraction = bt_cfg.get("bet_fraction", 0.02)
        self.kelly_fraction = bt_cfg.get("kelly_fraction", 0.25)
        self.bet_strategy = bt_cfg.get("bet_strategy", "kelly")

        self.results: list = []
        self.summary: dict = {}

    def implied_odds(self, prob: float) -> float:
        """Convert probability to decimal odds."""
        if prob <= 0 or prob >= 1:
            return 1.0
        return 1.0 / prob

    def kelly_bet_size(
        self,
        model_prob: float,
        market_prob: float,
        bankroll: float,
    ) -> float:
        """
        Compute Kelly criterion bet size.

        Kelly fraction f* = (bp - q) / b
        where:
        - b = decimal odds - 1 (net payout per unit wagered)
        - p = model probability of winning
        - q = 1 - p

        We use fractional Kelly (kelly_fraction) for risk management.
        """
        decimal_odds = self.implied_odds(market_prob)
        b = decimal_odds - 1.0
        if b <= 0:
            return 0.0

        p = model_prob
        q = 1.0 - p
        kelly_f = (b * p - q) / b

        # Fractional Kelly and floor at 0
        kelly_f = max(0.0, kelly_f * self.kelly_fraction)

        return bankroll * kelly_f

    def flat_bet_size(self, bankroll: float) -> float:
        """Compute flat bet size as a fraction of current bankroll."""
        return bankroll * self.bet_fraction

    def run(
        self,
        df: pd.DataFrame,
        model_prob_col: str = "model_prob_a",
        market_prob_col: str = "market_prob_a",
    ) -> pd.DataFrame:
        """
        Run the backtest simulation.

        Args:
            df: DataFrame with columns:
                - model_prob_a: model's predicted probability for team_a
                - market_prob_a: market-implied probability for team_a
                - actual_win_a: 1 if team_a won, 0 otherwise
                - team_a, team_b: team names
                - date (optional)

        Returns:
            DataFrame of all bets placed with outcomes and bankroll history.
        """
        df = df.copy()

        # Ensure required columns exist
        for col in [model_prob_col, market_prob_col, "actual_win_a"]:
            if col not in df.columns:
                raise ValueError(f"Required column '{col}' not found")

        bankroll = self.starting_bankroll
        self.results = []

        for idx, row in df.iterrows():
            model_prob = row[model_prob_col]
            market_prob = row[market_prob_col]
            actual = row["actual_win_a"]

            # Compute edge
            edge = model_prob - market_prob

            # Check both sides: bet on team_a if edge > threshold,
            # bet on team_b if edge < -threshold
            if abs(edge) < self.edge_threshold:
                continue

            # Determine bet side
            if edge > 0:
                # Bet on team_a
                bet_prob = model_prob
                bet_market_prob = market_prob
                bet_won = actual == 1.0
                bet_team = row.get("team_a", "team_a")
            else:
                # Bet on team_b
                bet_prob = 1.0 - model_prob
                bet_market_prob = 1.0 - market_prob
                bet_won = actual == 0.0
                bet_team = row.get("team_b", "team_b")

            # Compute bet size
            if self.bet_strategy == "kelly":
                bet_size = self.kelly_bet_size(bet_prob, bet_market_prob, bankroll)
            else:
                bet_size = self.flat_bet_size(bankroll)

            if bet_size <= 0:
                continue

            # Cap bet at current bankroll
            bet_size = min(bet_size, bankroll)

            # Compute payout
            decimal_odds = self.implied_odds(bet_market_prob)
            if bet_won:
                pnl = bet_size * (decimal_odds - 1.0)
            else:
                pnl = -bet_size

            bankroll += pnl

            self.results.append({
                "game_idx": idx,
                "date": row.get("date"),
                "team_a": row.get("team_a", ""),
                "team_b": row.get("team_b", ""),
                "bet_on": bet_team,
                "model_prob": bet_prob,
                "market_prob": bet_market_prob,
                "edge": abs(edge),
                "decimal_odds": decimal_odds,
                "bet_size": bet_size,
                "bet_won": bet_won,
                "pnl": pnl,
                "bankroll": bankroll,
            })

            # Stop if bankrupt
            if bankroll <= 0:
                break

        return pd.DataFrame(self.results)

    def compute_summary(self, results_df: Optional[pd.DataFrame] = None) -> dict:
        """
        Compute summary statistics for the backtest.

        Returns:
            Dict with key performance metrics.
        """
        if results_df is None:
            results_df = pd.DataFrame(self.results)

        if results_df.empty:
            self.summary = {
                "total_bets": 0,
                "message": "No bets placed (no edges found above threshold)",
            }
            return self.summary

        total_bets = len(results_df)
        wins = results_df["bet_won"].sum()
        losses = total_bets - wins

        total_pnl = results_df["pnl"].sum()
        final_bankroll = results_df["bankroll"].iloc[-1]
        roi = (final_bankroll - self.starting_bankroll) / self.starting_bankroll

        # Win rate
        win_rate = wins / total_bets if total_bets > 0 else 0

        # Average edge on bets
        avg_edge = results_df["edge"].mean()

        # Max drawdown
        cummax = results_df["bankroll"].cummax()
        drawdown = (results_df["bankroll"] - cummax) / cummax
        max_drawdown = drawdown.min()

        # Profit factor
        gross_profit = results_df[results_df["pnl"] > 0]["pnl"].sum()
        gross_loss = abs(results_df[results_df["pnl"] < 0]["pnl"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Sharpe-like ratio (daily returns)
        if "date" in results_df.columns and results_df["date"].notna().any():
            daily_pnl = results_df.groupby("date")["pnl"].sum()
            if daily_pnl.std() > 0:
                sharpe = (daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252)
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        self.summary = {
            "total_bets": total_bets,
            "wins": int(wins),
            "losses": int(losses),
            "win_rate": float(win_rate),
            "total_pnl": float(total_pnl),
            "roi": float(roi),
            "final_bankroll": float(final_bankroll),
            "starting_bankroll": float(self.starting_bankroll),
            "avg_edge": float(avg_edge),
            "max_drawdown": float(max_drawdown),
            "profit_factor": float(profit_factor),
            "sharpe_ratio": float(sharpe),
            "strategy": self.bet_strategy,
            "edge_threshold": self.edge_threshold,
        }
        return self.summary

    def print_summary(self, summary: Optional[dict] = None):
        """Print a formatted backtest summary."""
        s = summary or self.summary
        if not s:
            print("No summary available. Run backtest first.")
            return

        print("\n" + "=" * 55)
        print("  BACKTEST RESULTS")
        print("=" * 55)

        if s.get("total_bets", 0) == 0:
            print(f"  {s.get('message', 'No bets placed.')}")
            return

        print(f"  Strategy:        {s['strategy']} betting")
        print(f"  Edge Threshold:  {s['edge_threshold']:.1%}")
        print(f"  Total Bets:      {s['total_bets']}")
        print(f"  Win Rate:        {s['win_rate']:.1%} ({s['wins']}W / {s['losses']}L)")
        print(f"  Avg Edge:        {s['avg_edge']:.2%}")
        print("-" * 55)
        print(f"  Starting Bank:   ${s['starting_bankroll']:,.2f}")
        print(f"  Final Bank:      ${s['final_bankroll']:,.2f}")
        print(f"  Total P&L:       ${s['total_pnl']:+,.2f}")
        print(f"  ROI:             {s['roi']:+.1%}")
        print(f"  Max Drawdown:    {s['max_drawdown']:.1%}")
        print(f"  Profit Factor:   {s['profit_factor']:.2f}")
        print(f"  Sharpe Ratio:    {s['sharpe_ratio']:.2f}")
        print("=" * 55)

    def plot_bankroll(
        self,
        results_df: Optional[pd.DataFrame] = None,
        output_path: str = "backtest_bankroll.html",
    ):
        """Plot bankroll over time using plotly."""
        if not HAS_PLOTLY:
            print("plotly not installed. pip install plotly")
            return

        if results_df is None:
            results_df = pd.DataFrame(self.results)

        if results_df.empty:
            print("No results to plot.")
            return

        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=("Bankroll Over Time", "Per-Bet P&L"),
            vertical_spacing=0.12,
        )

        # Bankroll curve
        x_axis = results_df["date"] if "date" in results_df.columns and results_df["date"].notna().any() else range(len(results_df))
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=results_df["bankroll"],
                mode="lines",
                name="Bankroll",
                line=dict(color="blue", width=2),
            ),
            row=1, col=1,
        )

        # Starting bankroll reference
        fig.add_hline(
            y=self.starting_bankroll, line_dash="dash",
            line_color="gray", row=1, col=1,
        )

        # Per-bet P&L
        colors = ["green" if pnl > 0 else "red" for pnl in results_df["pnl"]]
        fig.add_trace(
            go.Bar(
                x=x_axis,
                y=results_df["pnl"],
                marker_color=colors,
                name="P&L",
            ),
            row=2, col=1,
        )

        fig.update_layout(height=700, showlegend=False, title_text="Backtest Results")
        fig.write_html(output_path)
        print(f"Backtest plot saved to {output_path}")

    def plot_edge_distribution(
        self,
        results_df: Optional[pd.DataFrame] = None,
        output_path: str = "backtest_edge_dist.html",
    ):
        """Plot distribution of edges on bets placed, colored by outcome."""
        if not HAS_PLOTLY:
            return

        if results_df is None:
            results_df = pd.DataFrame(self.results)

        if results_df.empty:
            return

        wins = results_df[results_df["bet_won"]]
        losses = results_df[~results_df["bet_won"]]

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=wins["edge"], name="Wins", marker_color="green", opacity=0.7))
        fig.add_trace(go.Histogram(x=losses["edge"], name="Losses", marker_color="red", opacity=0.7))

        fig.update_layout(
            barmode="overlay",
            title="Edge Distribution (Wins vs Losses)",
            xaxis_title="Edge (|model - market|)",
            yaxis_title="Count",
        )
        fig.write_html(output_path)
        print(f"Edge distribution plot saved to {output_path}")


if __name__ == "__main__":
    print("=" * 50)
    print("Backtest — Demo")
    print("=" * 50)

    np.random.seed(42)
    n = 300

    # Simulate: model has a small genuine edge over market
    market_probs = np.random.uniform(0.35, 0.65, n)
    # Model adds noise + small signal
    model_probs = market_probs + np.random.normal(0.02, 0.08, n)
    model_probs = np.clip(model_probs, 0.05, 0.95)

    # Outcomes more correlated with model than market
    true_probs = 0.6 * model_probs + 0.4 * market_probs
    outcomes = (np.random.random(n) < true_probs).astype(float)

    df = pd.DataFrame({
        "team_a": [f"Team_{i % 8}" for i in range(n)],
        "team_b": [f"Team_{(i + 4) % 8}" for i in range(n)],
        "model_prob_a": model_probs,
        "market_prob_a": market_probs,
        "actual_win_a": outcomes,
        "date": pd.date_range("2024-01-01", periods=n, freq="2D"),
    })

    bt = Backtest.__new__(Backtest)
    cfg = load_config()
    bt_cfg = cfg.get("backtest", {})
    bt.edge_threshold = bt_cfg.get("edge_threshold", 0.05)
    bt.starting_bankroll = bt_cfg.get("starting_bankroll", 1000.0)
    bt.bet_fraction = bt_cfg.get("bet_fraction", 0.02)
    bt.kelly_fraction = bt_cfg.get("kelly_fraction", 0.25)
    bt.bet_strategy = bt_cfg.get("bet_strategy", "kelly")
    bt.results = []
    bt.summary = {}

    results = bt.run(df)
    summary = bt.compute_summary(results)
    bt.print_summary(summary)
