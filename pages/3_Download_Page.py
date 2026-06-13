import os
import pandas as pd
import streamlit as st

from core.packager import create_zip
from core.config_user import COPYRIGHT_TEXT, PROJECT_TITLE

st.set_page_config(page_title="Download", layout="wide")
st.title(f"📥 Download: {PROJECT_TITLE}")
st.sidebar.markdown("---")
st.sidebar.info(COPYRIGHT_TEXT)

if "results" not in st.session_state or not st.session_state.results:
    st.warning("Please complete training first.")
    st.stop()

df = pd.DataFrame(st.session_state.results).copy()

# Pick best score column (macro preferred)
candidates = [
    "F1 (Macro)",
    "F1_macro",
    "F1-Score",
    "F1",
    "F1 (Weighted)",
    "F1_weighted",
    "Accuracy",
]

score_col = next((c for c in candidates if c in df.columns), None)

if score_col is None:
    st.error("No score column found to select the best model.")
    st.write("Available columns:", list(df.columns))
    st.stop()

df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
df_valid = df.dropna(subset=[score_col])

if df_valid.empty:
    st.error(f"Score column '{score_col}' has no valid numeric values.")
    st.stop()

df_pick = df_valid
if "Split" in df_valid.columns:
    # Prefer deploying the *_final model (trained on all data) if it exists.
    df_final = df_valid[df_valid["Split"].astype(str).isin(["final", "all"])].copy()
    if not df_final.empty:
        df_pick = df_final
        st.info("Deploy selection prefers **final/all** rows (models trained on all data), when available.")

best = df_pick.loc[df_pick[score_col].idxmax()]
algo = best.get("Algorithm")

if not algo:
    st.error("Missing 'Algorithm' field in results; cannot package model.")
    st.write(best)
    st.stop()

st.success(f"🏆 Best Model: **{algo}** ({score_col}: {best[score_col]:.4f})")

zip_path = create_zip(algo)

if not zip_path or not os.path.exists(zip_path):
    st.error("Deployment zip not found. Please re-run training and try again.")
    st.stop()

with open(zip_path, "rb") as f:
    st.download_button(
        label=f"⬇️ Download Best Model ({algo})",
        data=f,
        file_name=os.path.basename(zip_path),
        mime="application/zip",
    )

