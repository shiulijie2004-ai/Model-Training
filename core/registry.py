from core.config_user import CUSTOM_ALGOS

FULL_ALGO_MAP = {
    "RandomForest": "sklearn_tree",
    "SVM": "sklearn_linear",
    "DNN_Dense": "dnn",
    "CNN_1D": "cnn",
    "LSTM": "rnn",
    "KNN": "sklearn_linear",
    "DecisionTree": "sklearn_tree",
    "NaiveBayes": "sklearn_linear",
    "XGBoost": "xgboost",
    "XGBoost-Tiny": "xgboost_tiny",
}

# Inject user custom algos (default = tree)
for algo in CUSTOM_ALGOS:
    if algo not in FULL_ALGO_MAP:
        FULL_ALGO_MAP[algo] = "sklearn_tree"

CATEGORIES = {
    "My Selected Algorithms": CUSTOM_ALGOS,
    "Core Requirements": ["RandomForest", "SVM", "DNN_Dense", "CNN_1D", "LSTM","XGBoost"],
    "Full Library": list(FULL_ALGO_MAP.keys()),
}


