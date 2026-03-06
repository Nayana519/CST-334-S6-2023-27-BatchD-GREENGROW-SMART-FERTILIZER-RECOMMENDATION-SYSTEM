"""
Run this script ONCE to generate a clean dataset.csv that replaces
your original broken one. It produces 8000 rows with the exact same
columns your app and train_model.py expect, but with correct labels.

Usage:
    python3 generate_dataset.py
    python3 train_model.py
    python3 app.py
"""

import pandas as pd
import numpy as np

np.random.seed(42)
N_ROWS = 8000

# ── Crop and soil options (same as original dataset) ─────────────────────────
CROPS = ['Maize', 'Sugarcane', 'Cotton', 'Tobacco', 'Paddy',
         'Barley', 'Wheat', 'Millets', 'Oil seeds', 'Pulses', 'Ground Nuts']

SOILS = ['Sandy', 'Loamy', 'Black', 'Red', 'Clayey']

# ── Crop-specific optimal NPK (agronomic reference values) ───────────────────
OPTIMAL_NPK = {
    'Maize':       (35, 26, 30),
    'Sugarcane':   (25, 30, 20),
    'Cotton':      (20, 20, 20),
    'Tobacco':     (30, 25, 25),
    'Paddy':       (35, 20, 20),
    'Barley':      (25, 25, 20),
    'Wheat':       (40, 20, 20),
    'Millets':     (20, 20, 15),
    'Oil seeds':   (15, 25, 15),
    'Pulses':      (10, 30, 10),
    'Ground Nuts': (15, 30, 15),
}

# ── Crop-specific realistic climate ranges ────────────────────────────────────
CROP_CLIMATE = {
    'Maize':       dict(temp=(18, 35), humidity=(50, 80), moisture=(25, 60)),
    'Sugarcane':   dict(temp=(24, 38), humidity=(60, 85), moisture=(40, 70)),
    'Cotton':      dict(temp=(25, 40), humidity=(40, 70), moisture=(20, 50)),
    'Tobacco':     dict(temp=(20, 35), humidity=(45, 75), moisture=(25, 55)),
    'Paddy':       dict(temp=(22, 35), humidity=(60, 85), moisture=(50, 70)),
    'Barley':      dict(temp=(15, 30), humidity=(40, 70), moisture=(20, 50)),
    'Wheat':       dict(temp=(15, 28), humidity=(40, 70), moisture=(20, 50)),
    'Millets':     dict(temp=(25, 40), humidity=(40, 65), moisture=(20, 45)),
    'Oil seeds':   dict(temp=(20, 35), humidity=(45, 70), moisture=(20, 50)),
    'Pulses':      dict(temp=(18, 32), humidity=(45, 75), moisture=(25, 55)),
    'Ground Nuts': dict(temp=(22, 36), humidity=(45, 75), moisture=(25, 55)),
}

# ── Fertilizer NPK composition ────────────────────────────────────────────────
FERT_NPK = {
    'Urea':     np.array([46,  0,  0]),
    'DAP':      np.array([18, 46,  0]),
    '14-35-14': np.array([14, 35, 14]),
    '17-17-17': np.array([17, 17, 17]),
    '20-20':    np.array([20,  0, 20]),
    '28-28':    np.array([28,  0, 28]),
    '10-26-26': np.array([10, 26, 26]),
}

def best_fertilizer(n, p, k, crop):
    opt = OPTIMAL_NPK[crop]
    deficit = np.array([
        max(0, opt[0] - n),
        max(0, opt[1] - p),
        max(0, opt[2] - k),
    ], dtype=float)
    return min(FERT_NPK, key=lambda f: np.sum((deficit - FERT_NPK[f]) ** 2))

# ── Generate rows ─────────────────────────────────────────────────────────────
rows = []
for _ in range(N_ROWS):
    crop = np.random.choice(CROPS)
    soil = np.random.choice(SOILS)
    opt  = OPTIMAL_NPK[crop]
    clim = CROP_CLIMATE[crop]

    # Temperature, humidity, moisture — realistic ranges per crop
    temp     = round(np.random.uniform(*clim['temp']),     2)
    humidity = round(np.random.uniform(*clim['humidity']), 2)
    moisture = round(np.random.uniform(*clim['moisture']), 2)
    pH       = round(np.random.uniform(5.5, 7.5),          2)

    # NPK: sample current soil values below the crop optimum so there's always
    # a real deficit to fill. Add occasional excess to create variety.
    n = int(np.clip(np.random.normal(opt[0] * 0.6, opt[0] * 0.3), 0, opt[0] + 10))
    p = int(np.clip(np.random.normal(opt[1] * 0.6, opt[1] * 0.3), 0, opt[1] + 10))
    k = int(np.clip(np.random.normal(opt[2] * 0.6, opt[2] * 0.3), 0, opt[2] + 10))

    fertilizer = best_fertilizer(n, p, k, crop)

    rows.append({
        'Temparature':      temp,      # keep original typo to match app.py
        'Humidity':         humidity,
        'Moisture':         moisture,
        'Soil Type':        soil,
        'Crop Type':        crop,
        'Nitrogen':         n,
        'Potassium':        k,
        'Phosphorous':      p,
        'Fertilizer Name':  fertilizer,
        'pH':               pH,
    })

df = pd.DataFrame(rows)

# ── Sanity checks ─────────────────────────────────────────────────────────────
print("=== Generated Dataset ===")
print(f"Shape : {df.shape}")
print(f"\nFertilizer distribution:")
print(df['Fertilizer Name'].value_counts().to_string())
print(f"\nCrop distribution:")
print(df['Crop Type'].value_counts().to_string())
print(f"\nNPK ranges:")
print(df[['Nitrogen', 'Phosphorous', 'Potassium']].describe().round(1).to_string())
print(f"\nSample rows:")
print(df.head(10).to_string())

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("dataset.csv", index=False)
print("\n✅ dataset.csv saved — now run: python3 train_model.py")