import pandas as pd

df = pd.read_csv('dataset.csv')
df.columns = [c.strip() for c in df.columns]

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

LOCATIONS = {
    'Maize':       'Dharwad, Karnataka',
    'Sugarcane':   'Kolhapur, Maharashtra',
    'Cotton':      'Amreli, Gujarat',
    'Tobacco':     'Guntur, Andhra Pradesh',
    'Paddy':       'Thanjavur, Tamil Nadu',
    'Barley':      'Jaipur, Rajasthan',
    'Wheat':       'Ludhiana, Punjab',
    'Millets':     'Jodhpur, Rajasthan',
    'Oil seeds':   'Kota, Rajasthan',
    'Pulses':      'Vidisha, Madhya Pradesh',
    'Ground Nuts': 'Anantapur, Andhra Pradesh',
}

def assign_fertilizer(row):
    opt = OPTIMAL_NPK.get(row['Crop Type'], (20, 20, 20))
    dn = max(0, opt[0] - row['Nitrogen'])
    dp = max(0, opt[1] - row['Phosphorous'])
    dk = max(0, opt[2] - row['Potassium'])
    total = dn + dp + dk
    if total == 0:
        return 'Iffco NPK 17:17:17'
    pn, pp, pk = dn/total, dp/total, dk/total
    if pn >= 0.55 and dn >= 8:
        return 'Urea (46% N)'
    if pp >= 0.50 and dp >= 8 and dn >= 2:
        return 'DAP (18:46:0)'
    if pn <= 0.22 and (pp+pk) >= 0.78 and (dp+dk) >= 10:
        return 'Tata Paras NPK 10:26:26'
    if pp >= 0.38 and pk >= 0.14 and pp > pn:
        return 'NPK Complex 14:35:14'
    if pn >= 0.35 and pk >= 0.28 and pp <= 0.25:
        return 'Kribhco NPK 20:20:0'
    if pn >= 0.28 and pp >= 0.25 and pk <= 0.30 and abs(pn-pp) <= 0.15:
        return 'MAP 28:28:0'
    return 'Iffco NPK 17:17:17'

df['Model_Prediction'] = df.apply(assign_fertilizer, axis=1)

crops = df['Crop Type'].unique().tolist()
rows = []
for crop in crops:
    sub = df[df['Crop Type'] == crop].sample(min(2, len(df[df['Crop Type'] == crop])), random_state=7)
    rows.append(sub)

import pandas as pd
samples = pd.concat(rows).reset_index(drop=True)

with open('proof_rows.txt', 'w') as f:
    for i, row in samples.iterrows():
        crop = row['Crop Type']
        opt = OPTIMAL_NPK.get(crop, (20,20,20))
        dn = max(0, opt[0] - row['Nitrogen'])
        dp = max(0, opt[1] - row['Phosphorous'])
        dk = max(0, opt[2] - row['Potassium'])
        match = "YES" if row['Fertilizer Name'] == row['Model_Prediction'] else "NO"
        line = (crop + "|" + row['Soil Type'] + "|" +
              str(int(row['Nitrogen'])) + "|" +
              str(int(row['Phosphorous'])) + "|" +
              str(int(row['Potassium'])) + "|" +
              str(round(float(row['pH']), 2)) + "|" +
              str(round(float(row['Temparature']), 1)) + "|" +
              str(round(float(row['Humidity']), 1)) + "|" +
              str(round(float(row['Moisture']), 1)) + "|" +
              LOCATIONS.get(crop, 'India') + "|" +
              row['Fertilizer Name'] + "|" +
              row['Model_Prediction'] + "|" +
              str(opt[0]) + "|" + str(opt[1]) + "|" + str(opt[2]) + "|" +
              str(dn) + "|" + str(dp) + "|" + str(dk) + "|" +
              match)
        f.write(line + "\n")

print("Saved to proof_rows.txt")
