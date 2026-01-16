import os
import pandas as pd
import streamlit as st

from core.registry import CATEGORIES
from core.trainer import StudentTrainer
from core.profiler import EdgeProfiler
from core.config_user import COPYRIGHT_TEXT, PROJECT_TITLE

st.set_page_config(page_title="Training", layout="wide")
st.title(f"Training Lab: {PROJECT_TITLE}")

st.sidebar.markdown("---")
st.sidebar.info(COPYRIGHT_TEXT)

if "results" not in st.session_state:
    st.session_state.results = []

profiler = EdgeProfiler()

with st.sidebar:
    st.header("1) Target Device")
    device = st.selectbox("Hardware", profiler.get_device_list())

    st.header("2) Dataset")
    datasets = sorted([f for f in os.listdir("data") if f.lower().endswith(".csv")])
    if not datasets:
        st.error("No CSV files found in the data/ folder.")
        st.stop()
    dataset = st.selectbox("Data", datasets)

    df_tmp = pd.read_csv(f"data/{dataset}")
    st.caption("Note: Labels will be auto-mapped into 4 classes: eating / ruminating / standing / walking.")

    st.header("3) Algorithms")
    all_models = []
    if "My Selected Algorithms" in CATEGORIES:
        all_models += CATEGORIES["My Selected Algorithms"]
    all_models += CATEGORIES.get("Core Requirements", [])
    all_models += CATEGORIES.get("Full Library", [])
    all_models = list(dict.fromkeys(all_models))
    selected = st.multiselect("Select Algorithms", all_models)

    st.header("4) Split")
    split_method = st.selectbox(
        "Split method",
        ["random", "time", "group", "loco", "full_loco", "all"],
        help=(
            "random/time: window-wise split (can leak animal identity).\n"
            "group: group-wise split (no identity leakage).\n"
            "loco: leave-one-animal-out for a SINGLE selected animal.\n"
            "full_loco: leave-one-animal-out for EVERY animal + also trains a final deployment model.\n"
            "all: no split; train on the full dataset and output *_final.*"
        ),
    )

    candidate_cols = ["cow_id", "cow_num", "calfId", "calf_id", "animal_id", "animalId", "subject", "id"]
    guess = ""
    for c in candidate_cols:
        if c in df_tmp.columns:
            guess = c
            break

    group_col = ""
    test_group = None

    # ✅ FIX: full_loco also needs group_col, so include it here
    if split_method in ["group", "loco", "full_loco"]:
        group_col = st.text_input("Group column (e.g., cow_id / cow_num / calfId)", value=guess).strip()

        if not group_col:
            st.warning("This split method requires a group column.")
        elif group_col not in df_tmp.columns:
            st.warning(f"Group column not found in dataset: {group_col}")
        else:
            groups = sorted(df_tmp[group_col].astype(str).unique().tolist())
            if split_method == "loco":
                test_group = st.selectbox("Leave-out test group", groups)
            elif split_method == "full_loco":
                st.info(
                    f"FULL LOCO will evaluate all {len(groups)} groups (one-by-one) "
                    "and then train a final '*_final' deployment model on ALL data."
                )

    st.header("5) Windowing")
    seq_window = st.slider("Sequence window (CNN/LSTM timesteps)", 1, 200, 50, step=5)

    use_classic_window = st.checkbox(
        "Apply window statistics to classic ML (RF / SVM / XGBoost) — heavier but sometimes improves stability",
        value=False
    )
    classic_window = 1
    if use_classic_window:
        classic_window = st.slider("Classic window (stats window)", 1, 200, 50, step=5)

    st.header("6) Tuning")
    hp_epochs = st.slider("Epochs (DL)", 1, 30, 5)
    hp_batch = st.select_slider("Batch Size", [8, 16, 32, 64, 128], value=32)
    hp_est = st.slider("Estimators (RF)", 10, 300, 50)

    start_btn = st.button("START TRAINING")

if start_btn:
    if not selected:
        st.error("Select at least one algorithm.")
    else:
        trainer = StudentTrainer()
        st.session_state.results = []
        bar = st.progress(0.0)

        total_jobs = len(selected)
        for i, algo in enumerate(selected):
            try:
                res = trainer.train_model(
                    algo_name=algo,
                    data_path=f"data/{dataset}",
                    device_name=device,
                    hp_epochs=hp_epochs,
                    hp_batch=hp_batch,
                    hp_est=hp_est,
                    window=seq_window,
                    classic_window=classic_window,
                    split_method=split_method,
                    group_col=group_col if group_col else None,
                    test_group=test_group,
                )
                # full_loco returns a list of dicts (one per group) + a final model row
                if isinstance(res, list):
                    st.session_state.results.extend(res)
                else:
                    st.session_state.results.append(res)
            except Exception as e:
                st.error(f"Error {algo}: {e}")

            bar.progress(min(1.0, (i + 1) / max(1, total_jobs)))

        st.success("Training completed.")

if st.session_state.results:
    st.write(f"Results for device: {device}")
    st.dataframe(pd.DataFrame(st.session_state.results), use_container_width=True)

