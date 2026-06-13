import os
import numpy as np
import pandas as pd

os.makedirs("data", exist_ok=True)

def generate(name, n, features, logic):
    data = {k: np.random.normal(mu, sigma, n) for k, (mu, sigma) in features.items()}
    df = pd.DataFrame(data)
    df["Target"] = logic(df).astype(int)
    df.to_csv(f"data/{name}.csv", index=False)

# Binary toy datasets for demo
generate(
    "Industrial_Motor",
    3000,
    {"Vib_X": (0.5, 0.1), "Temp": (65, 5)},
    lambda d: (d["Vib_X"] > 0.62) | (d["Temp"] > 72),
)
generate(
    "Smart_Grid",
    2500,
    {"Volt": (230, 5), "Freq": (50, 0.2)},
    lambda d: (d["Volt"] < 222) | (d["Volt"] > 238) | (d["Freq"] < 49.6) | (d["Freq"] > 50.4),
)
generate(
    "Agri_Sensor",
    2000,
    {"Moisture": (40, 10), "PH": (6.5, 0.5)},
    lambda d: (d["Moisture"] < 25) | (d["Moisture"] > 60) | (d["PH"] < 5.8) | (d["PH"] > 7.4),
)
print("[ok] generated demo CSV datasets in ./data/")
