from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

RESULT_LABELS = ["Away win", "Draw", "Home win"]


def load_model_and_columns():
    model = joblib.load(MODELS_DIR / "xgboost_v2.pkl")
    feature_cols = joblib.load(MODELS_DIR / "feature_columns_v2.pkl")
    return model, feature_cols


def load_history() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_DIR / "model_features.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def latest_team_stats(team: str, df: pd.DataFrame) -> dict:
    """Most recent known rank/points/form for a team, whether they last
    played as home or away."""
    team_matches = df[(df["home_team"] == team) | (df["away_team"] == team)]
    if team_matches.empty:
        raise ValueError(
            f"No historical data found for '{team}'. Check spelling matches "
            f"dataset's team names exactly (e.g. 'USA' vs 'United States')."
        )
    last = team_matches.sort_values("date").iloc[-1]
    prefix = "home" if last["home_team"] == team else "away"

    return {
        "rank": last.get(f"{prefix}_rank", np.nan),
        "points": last.get(f"{prefix}_points", np.nan),
        "form_win_rate": last.get(f"{prefix}_form_win_rate", np.nan),
        "form_goals_for": last.get(f"{prefix}_form_goals_for", last.get(f"{prefix}_form_goal_for", np.nan)),
        "form_goals_against": last.get(f"{prefix}_form_goals_against", last.get(f"{prefix}_form_goal_against", np.nan)),
        "last_match_date": last["date"],
    }


def head_to_head_rate(home_team: str, away_team: str, df: pd.DataFrame) -> float:
    """Historical fraction of meetings between these two teams that the
    'home_team' argument won (regardless of which side they were on
    historically). Returns 0.5 (neutral/unknown) if they've never met."""
    past = df[
        ((df["home_team"] == home_team) & (df["away_team"] == away_team))
        | ((df["home_team"] == away_team) & (df["away_team"] == home_team))
    ]
    if past.empty:
        return 0.5

    home_wins = 0
    for _, row in past.iterrows():
        if row["home_team"] == home_team and row["result_encoded"] == 2:
            home_wins += 1
        elif row["away_team"] == home_team and row["result_encoded"] == 0:
            home_wins += 1
    return home_wins / len(past)


def build_match_row(
    home_team: str,
    away_team: str,
    match_date: str,
    df: pd.DataFrame,
    feature_cols: list,
    tournament_bucket: str = "world_cup",
    neutral: bool = True,
) -> pd.DataFrame:
    home_stats = latest_team_stats(home_team, df)
    away_stats = latest_team_stats(away_team, df)
    match_ts = pd.Timestamp(match_date)

    row = {
        "neutral": neutral,
        "home_rank": home_stats["rank"],
        "home_points": home_stats["points"],
        "away_rank": away_stats["rank"],
        "away_points": away_stats["points"],
        "rank_diff": away_stats["rank"] - home_stats["rank"],
        "home_form_win_rate": home_stats["form_win_rate"],
        "home_form_goals_for": home_stats["form_goals_for"],
        "home_form_goals_against": home_stats["form_goals_against"],
        "home_form_goal_for": home_stats["form_goals_for"],       # alt naming
        "home_form_goal_against": home_stats["form_goals_against"],  # alt naming
        "away_form_win_rate": away_stats["form_win_rate"],
        "away_form_goals_for": away_stats["form_goals_for"],
        "away_form_goals_against": away_stats["form_goals_against"],
        "away_form_goal_for": away_stats["form_goals_for"],       # alt naming
        "away_form_goal_against": away_stats["form_goals_against"],  # alt naming
        "home_rest_days": (match_ts - home_stats["last_match_date"]).days,
        "away_rest_days": (match_ts - away_stats["last_match_date"]).days,
        "h2h_home_win_rate": head_to_head_rate(home_team, away_team, df),
    }

    for bucket in ["world_cup", "world_cup_qualifier", "continental_championship", "friendly", "other"]:
        row[f"tournament_bucket_{bucket}"] = 1 if bucket == tournament_bucket else 0

    # Only keep columns the trained model actually expects, in the exact
    # order it expects them. Anything missing defaults to 0 rather than
    # crashing — check the printed warning if that happens.
    input_row = {}
    for col in feature_cols:
        if col in row:
            input_row[col] = row[col]
        else:
            print(f"  WARNING: no value computed for feature '{col}', defaulting to 0")
            input_row[col] = 0

    return pd.DataFrame([input_row])


def predict_match(home_team: str, away_team: str, match_date: str, df, model, feature_cols) -> None:
    X_new = build_match_row(home_team, away_team, match_date, df, feature_cols)
    proba = model.predict_proba(X_new)[0]
    away_p, draw_p, home_p = proba  # order matches result_encoded: 0=Away, 1=Draw, 2=Home

    print(f"\n{home_team} vs {away_team} ({match_date})")
    print(f"  {home_team}: {home_p:.1%}")
    print(f"  Draw: {draw_p:.1%}")
    print(f"  {away_team}: {away_p:.1%}")

    outcomes = {home_team: home_p, "Draw": draw_p, away_team: away_p}
    predicted = max(outcomes, key=outcomes.get)
    print(f"  --> Predicted: {predicted}")


def main() -> None:
    model, feature_cols = load_model_and_columns()
    df = load_history()

    print(f"Model expects {len(feature_cols)} features: {feature_cols}")

    # ---- EDIT THESE for the actual upcoming fixtures ----
    predict_match("France", "Spain", "2026-07-15", df, model, feature_cols)
    predict_match("England", "Argentina", "2026-07-16", df, model, feature_cols)

    # Add the final once semifinal winners are known:
    print("Final\n")
    predict_match("Spain", "Argentina", "2026-07-19", df, model, feature_cols)


if __name__ == "__main__":
    main()