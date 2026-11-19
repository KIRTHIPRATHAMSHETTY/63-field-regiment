"""
BARBARIK - Scenario History page
================================

Shows every scenario stored in the SQLite database, with outcome filtering,
per-row deletion and a CSV export. All data comes from src/database.py.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from database import get_all_scenarios, delete_scenario


def _rows_to_df(rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": r["id"],
                "timestamp": r["timestamp"],
                "predicted_outcome": r["predicted_outcome"],
                "prob_a": r["prob_a"],
                "prob_b": r["prob_b"],
                "prob_draw": r["prob_draw"],
            }
            for r in rows
        ]
    )


def render() -> None:
    st.title("Scenario History")
    rows = get_all_scenarios()          # real DB read (empty -> [])

    if not rows:
        st.info("No scenarios recorded yet. Run a simulation first.")
        return

    df = _rows_to_df(rows)

    # -- Filter by outcome --------------------------------------------------
    st.markdown("#### Filter")
    outcomes = ["ALL"] + sorted(df["predicted_outcome"].unique().tolist())
    chosen = st.selectbox("Filter by outcome", outcomes)
    if chosen != "ALL":
        df_f = df[df["predicted_outcome"] == chosen]
    else:
        df_f = df

    st.markdown(f"Showing **{len(df_f)}** of **{len(df)}** scenarios.")

    # -- Deletion -----------------------------------------------------------
    delete_target = st.selectbox("Select scenario to delete", ["-"] + df["id"].tolist())
    if delete_target != "-":
        if st.button("DELETE SCENARIO", type="secondary"):
            delete_scenario(delete_target)
            st.success(f"Deleted {delete_target}.")
            st.rerun()

    # -- Display ------------------------------------------------------------
    display = df_f.copy()
    display["A%"] = (display["prob_a"] * 100).round(1)
    display["B%"] = (display["prob_b"] * 100).round(1)
    display["Draw%"] = (display["prob_draw"] * 100).round(1)
    st.dataframe(
        display[
            ["id", "timestamp", "predicted_outcome", "A%", "B%", "Draw%"]
        ],
        width="stretch",
        hide_index=True,
    )

    # -- Export -------------------------------------------------------------
    st.download_button(
        "EXPORT CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="barbarik_history.csv",
        mime="text/csv",
    )