from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import joblib
import requests
import os
from dotenv import load_dotenv
load_dotenv()
import numpy as np
import functools
import uuid
from PIL import Image
import io

# -------------------------------
# 1. INITIALIZE FLASK APP
# -------------------------------
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")

DB_PATH = "users.db"

UPLOAD_FOLDER = os.path.join(os.getcwd(), "static", "uploads", "avatars")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif"}
MAX_FILE_SIZE = 5 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE


def init_db():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            location TEXT,
            farm_size TEXT,
            crops TEXT,
            soil_type TEXT,
            avatar_filename TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )''')
        c.execute('''CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY,
            user_id INTEGER,
            nitrogen REAL,
            phosphorus REAL,
            potassium REAL,
            ph REAL,
            crop TEXT,
            soil TEXT,
            fertilizer_result TEXT,
            quantity_kg_per_ha REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )''')
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"⚠️ Database initialization error: {e}")


def migrate_db():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=5)
        c = conn.cursor()

        # Check and add avatar_filename column if missing
        c.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in c.fetchall()]
        if 'avatar_filename' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN avatar_filename TEXT")
            conn.commit()
            print("✅ Database migration: Added avatar_filename column")

        # Check and add quantity_kg_per_ha column if missing
        c.execute("PRAGMA table_info(predictions)")
        pred_columns = [col[1] for col in c.fetchall()]
        if 'quantity_kg_per_ha' not in pred_columns:
            c.execute("ALTER TABLE predictions ADD COLUMN quantity_kg_per_ha REAL")
            conn.commit()
            print("✅ Database migration: Added quantity_kg_per_ha column")

        conn.close()
    except Exception as e:
        print(f"⚠️ Database migration error: {e}")


init_db()
migrate_db()


def login_required(f):
    @functools.wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function


def get_user(user_id):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        user = c.fetchone()
        conn.close()
        return dict(user) if user else None
    except Exception as e:
        print(f"Error fetching user: {e}")
        return None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def optimize_image(image_path, max_width=200, max_height=200, quality=85):
    try:
        img = Image.open(image_path)
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        img.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
        img.save(image_path, 'JPEG', quality=quality, optimize=True)
        return True
    except Exception as e:
        print(f"Image optimization error: {e}")
        return False


def get_avatar_url(user_id):
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('SELECT avatar_filename FROM users WHERE id = ?', (user_id,))
        result = c.fetchone()
        conn.close()
        if result and result['avatar_filename']:
            return f"/static/uploads/avatars/{result['avatar_filename']}"
        return None
    except Exception as e:
        print(f"Error getting avatar URL: {e}")
        return None


# -------------------------------
# 2. LOAD MODEL + ENCODERS
# -------------------------------
model       = joblib.load("models/model.pkl")              if os.path.exists("models/model.pkl")              else None
le_soil     = joblib.load("models/soil_encoder.pkl")       if os.path.exists("models/soil_encoder.pkl")       else None
le_crop     = joblib.load("models/crop_encoder.pkl")       if os.path.exists("models/crop_encoder.pkl")       else None
le_fert     = joblib.load("models/fertilizer_encoder.pkl") if os.path.exists("models/fertilizer_encoder.pkl") else None
OPTIMAL_NPK = joblib.load("models/optimal_npk.pkl")        if os.path.exists("models/optimal_npk.pkl")        else {}

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_KEY")


# -------------------------------
# 3. WEATHER FUNCTION
# -------------------------------
def get_weather(city):
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        response = requests.get(url, timeout=5)
        data = response.json()
        if str(data.get("cod")) != "200":
            return 25.0, 50.0, "Weather Unavailable"
        temp = round(data["main"]["temp"], 1)
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"].title()
        return temp, humidity, description
    except Exception as e:
        print(f"[ERROR] Weather API failed: {e}")
        return 25.0, 50.0, "Weather Unavailable"


# -------------------------------
# 4. FERTILIZER QUANTITY CALCULATION
# -------------------------------

# Fertilizer NPK composition (% by weight)
FERT_NPK_COMPOSITION = {
    'Urea':     {'N': 46, 'P': 0,  'K': 0},
    'DAP':      {'N': 18, 'P': 46, 'K': 0},
    '14-35-14': {'N': 14, 'P': 35, 'K': 14},
    '17-17-17': {'N': 17, 'P': 17, 'K': 17},
    '20-20':    {'N': 20, 'P': 0,  'K': 20},
    '28-28':    {'N': 28, 'P': 0,  'K': 28},
    '10-26-26': {'N': 10, 'P': 26, 'K': 26},
}


def calculate_fertilizer_quantity(fertilizer_name, def_n, def_p, def_k, crop_name=""):
    """
    Calculate recommended fertilizer quantity based on NPK deficits using agronomic formulas.

    Formula:
    - For each nutrient deficit: Required Fertilizer = Deficit / (Nutrient % / 100)
    - Take the maximum requirement to ensure all deficits are covered
    - Add 10% buffer for application efficiency losses
    - Apply crop-specific adjustment factors
    - Round to nearest 5 kg/ha for practical field application

    Args:
        fertilizer_name: Name of the recommended fertilizer
        def_n: Nitrogen deficit (kg/ha) = Optimal N - Current N
        def_p: Phosphorus deficit (kg/ha) = Optimal P - Current P
        def_k: Potassium deficit (kg/ha) = Optimal K - Current K
        crop_name: Crop type for additional adjustments

    Returns:
        Quantity in kg/ha (rounded to nearest 5 kg)
    """
    try:
        if fertilizer_name not in FERT_NPK_COMPOSITION:
            return 100

        fert = FERT_NPK_COMPOSITION[fertilizer_name]
        n_pct = fert['N']
        p_pct = fert['P']
        k_pct = fert['K']

        required_amounts = []

        if n_pct > 0 and def_n > 0:
            required_amounts.append(def_n / (n_pct / 100))
        if p_pct > 0 and def_p > 0:
            required_amounts.append(def_p / (p_pct / 100))
        if k_pct > 0 and def_k > 0:
            required_amounts.append(def_k / (k_pct / 100))

        base_quantity = max(required_amounts) if required_amounts else 80

        # Crop-specific adjustment factors based on nutrient demand
        crop_factors = {
            'Sugarcane': 1.2,
            'Cotton':    1.15,
            'Tobacco':   1.1,
            'Paddy':     1.1,
            'Maize':     1.05,
            'Wheat':     1.0,
            'Barley':    0.95,
            'Millets':   0.9,
            'Pulses':    0.85,
        }

        crop_factor = crop_factors.get(crop_name, 1.0)
        adjusted_quantity = base_quantity * crop_factor

        # Add 10% buffer for application efficiency losses
        quantity_with_buffer = adjusted_quantity * 1.1

        # Apply safety min/max limits
        quantity_with_buffer = max(50, min(quantity_with_buffer, 500))

        # Round to nearest 5 kg/ha
        final_quantity = round(quantity_with_buffer / 5) * 5

        return int(final_quantity)

    except Exception as e:
        print(f"[ERROR] Error calculating fertilizer quantity: {e}")
        return 100


def get_application_instructions(fertilizer_name, quantity, crop_name):
    """
    Generate detailed application instructions based on fertilizer type and crop.

    Args:
        fertilizer_name: Name of the fertilizer
        quantity: Recommended quantity in kg/ha
        crop_name: Crop type

    Returns:
        Dictionary with application instructions
    """
    instructions = {
        'Urea': {
            'timing': 'Apply in 2-3 split doses during vegetative growth stages',
            'method': 'Broadcast application or top dressing',
            'precautions': 'Avoid applying during flowering. Apply when soil has adequate moisture.',
            'splits': f'Base dose: {int(quantity * 0.3)} kg/ha, First top dress: {int(quantity * 0.35)} kg/ha, Second top dress: {int(quantity * 0.35)} kg/ha'
        },
        'DAP': {
            'timing': 'Apply as basal dose before sowing or transplanting',
            'method': 'Mix with soil during land preparation or apply in furrows',
            'precautions': 'Ensure proper soil moisture. Place 5-7 cm deep near seed placement.',
            'splits': f'Single basal application: {quantity} kg/ha'
        },
        '14-35-14': {
            'timing': 'Apply as basal dose with first split for top dressing',
            'method': 'Broadcast and incorporate into soil',
            'precautions': 'Suitable for high phosphorus demand crops',
            'splits': f'Basal: {int(quantity * 0.6)} kg/ha, Top dress: {int(quantity * 0.4)} kg/ha'
        },
        '17-17-17': {
            'timing': 'Apply in 2 split doses - basal and top dressing',
            'method': 'Broadcast application with soil incorporation',
            'precautions': 'Balanced fertilizer suitable for all growth stages',
            'splits': f'Basal: {int(quantity * 0.5)} kg/ha, Top dress: {int(quantity * 0.5)} kg/ha'
        },
        '20-20': {
            'timing': 'Apply during vegetative to reproductive stage',
            'method': 'Broadcast application',
            'precautions': 'Good for nitrogen and potassium boost',
            'splits': f'Split into 2 doses: {int(quantity * 0.5)} kg/ha each'
        },
        '28-28': {
            'timing': 'Apply in split doses during active growth',
            'method': 'Broadcast or band application',
            'precautions': 'High nutrient concentration - use with care',
            'splits': f'Split into 2-3 doses of approximately {int(quantity / 2.5)} kg/ha each'
        },
        '10-26-26': {
            'timing': 'Apply as basal dose, suitable for potassium-loving crops',
            'method': 'Broadcast and mix with soil',
            'precautions': 'Excellent for fruit and tuber crops',
            'splits': f'Basal application: {quantity} kg/ha'
        }
    }

    default_instructions = {
        'timing': 'Apply in split doses during crop growth stages',
        'method': 'Broadcast application and incorporate into soil',
        'precautions': 'Follow recommended dosage and timing',
        'splits': f'Split into 2 doses: {int(quantity / 2)} kg/ha each'
    }

    return instructions.get(fertilizer_name, default_instructions)


# -------------------------------
# 5. ROUTES
# -------------------------------

@app.route("/")
def home():
    return render_template("landing.html")

# Keep /landing as an alias so any url_for('landing') in old templates still works
@app.route("/landing")
def landing():
    return redirect(url_for('home'))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm  = request.form.get("confirm", "").strip()

        # confirm field is optional — only validate if it was submitted
        if not all([name, email, password]):
            return render_template("register.html", error="All fields are required")
        if confirm and password != confirm:
            return render_template("register.html", error="Passwords don't match")

        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            c = conn.cursor()
            c.execute('SELECT id FROM users WHERE email=?', (email,))
            if c.fetchone():
                conn.close()
                return render_template("register.html", error="Email already registered")
            c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                      (name, email, generate_password_hash(password)))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Email already exists")
        except Exception as e:
            print(f"Registration error: {e}")
            return render_template("register.html", error="Database error occurred")

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()

        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            conn.row_factory = sqlite3.Row
            c = conn.cursor()
            c.execute('SELECT * FROM users WHERE email = ?', (email,))
            user = c.fetchone()
            conn.close()

            if user and check_password_hash(user['password'], password):
                session['user_id']   = user['id']
                session['user_name'] = user['name']
                return redirect(url_for('predict_page'))

            return render_template("login.html", error="Invalid email or password")
        except Exception as e:
            print(f"Database error during login: {e}")
            return render_template("login.html", error="Database error. Please try again later.")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for('home'))


@app.route("/api/auth-status", methods=["GET"])
def auth_status():
    if 'user_id' in session:
        user = get_user(session['user_id'])
        return jsonify({
            'authenticated': True,
            'user_id':   session['user_id'],
            'user_name': session.get('user_name', ''),
            'email':     user.get('email', '') if user else ''
        })
    return jsonify({'authenticated': False})


@app.route("/profile")
@login_required
def profile():
    user = get_user(session['user_id'])
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''SELECT * FROM predictions
                     WHERE user_id=?
                     ORDER BY created_at DESC
                     LIMIT 10''',
                  (session['user_id'],))
        predictions = [dict(row) for row in c.fetchall()]
        conn.close()
    except Exception as e:
        print(f"Error fetching predictions for profile: {e}")
        predictions = []
    return render_template("profile.html", user=user, predictions=predictions)


@app.route("/profile/update", methods=["POST"])
@login_required
def profile_update():
    name      = request.form.get("name", "").strip()
    location  = request.form.get("location", "").strip()
    farm_size = request.form.get("farm_size", "").strip()
    crops     = request.form.get("crops", "").strip()
    soil_type = request.form.get("soil_type", "").strip()

    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute('''UPDATE users SET name=?, location=?, farm_size=?, crops=?, soil_type=?
                     WHERE id=?''',
                  (name, location, farm_size, crops, soil_type, session['user_id']))
        conn.commit()
        conn.close()
        session['user_name'] = name
        return redirect(url_for('profile'))
    except Exception as e:
        print(f"Error updating profile: {e}")
        return redirect(url_for('profile'))


@app.route("/profile/upload-avatar", methods=["POST"])
@login_required
def upload_avatar():
    try:
        if 'avatar' not in request.files:
            return redirect(url_for('profile'))
        file = request.files['avatar']
        if file.filename == '' or not allowed_file(file.filename):
            return redirect(url_for('profile'))
        file.seek(0, 2)
        if file.tell() > MAX_FILE_SIZE:
            return redirect(url_for('profile'))
        file.seek(0)

        filename = f"{session['user_id']}_{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        optimize_image(filepath)

        user = get_user(session['user_id'])
        if user and user.get('avatar_filename'):
            old = os.path.join(UPLOAD_FOLDER, user['avatar_filename'])
            try:
                if os.path.exists(old):
                    os.remove(old)
            except Exception:
                pass

        conn = sqlite3.connect(DB_PATH, timeout=10)
        c = conn.cursor()
        c.execute('UPDATE users SET avatar_filename=? WHERE id=?',
                  (filename, session['user_id']))
        conn.commit()
        conn.close()
        return redirect(url_for('profile'))
    except Exception as e:
        print(f"Error uploading avatar: {e}")
        return redirect(url_for('profile'))


@app.route("/predict-page")
@login_required
def predict_page():
    return render_template("index.html")


@app.route("/history")
@login_required
def history():
    try:
        conn = sqlite3.connect(DB_PATH, timeout=10)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute('''SELECT * FROM predictions
                     WHERE user_id=?
                     ORDER BY created_at DESC
                     LIMIT 20''',
                  (session['user_id'],))
        predictions_raw = c.fetchall()
        conn.close()

        # Convert to list of dicts for easier template handling
        predictions = [dict(row) for row in predictions_raw]

        # Recalculate quantity for any rows where it's missing
        for pred in predictions:
            if pred.get('quantity_kg_per_ha') is None or pred.get('quantity_kg_per_ha') == 0:
                crop_name = pred.get('crop', 'Wheat')
                opt = OPTIMAL_NPK.get(crop_name, (20, 20, 20))
                def_n = max(0, opt[0] - (pred.get('nitrogen')   or 0))
                def_p = max(0, opt[1] - (pred.get('phosphorus') or 0))
                def_k = max(0, opt[2] - (pred.get('potassium')  or 0))
                pred['quantity_kg_per_ha'] = calculate_fertilizer_quantity(
                    pred.get('fertilizer_result', 'Urea'),
                    def_n, def_p, def_k,
                    crop_name
                )

        return render_template("history.html", predictions=predictions)
    except Exception as e:
        print(f"Error fetching history: {e}")
        return render_template("history.html", predictions=[], error=str(e))


# -------------------------------
# 6. PREDICTION ROUTE
# -------------------------------
@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        if model is None:
            return "❌ model.pkl not found. Run train_model.py first.", 500

        # Get form data
        N         = float(request.form.get("Nitrogen",   0))
        P         = float(request.form.get("Phosphorus", 0))
        K         = float(request.form.get("Potassium",  0))
        pH        = float(request.form.get("pH",         6.5))
        moisture  = float(request.form.get("Moisture",   50.0))
        crop_name = request.form.get("Crop",  "Wheat")
        soil_name = request.form.get("Soil",  "Sandy")
        city      = request.form.get("City",  "Mumbai")

        # Weather
        temp, humidity, weather_desc = get_weather(city)

        # Encode soil
        try:
            soil_encoded = le_soil.transform([soil_name])[0] if le_soil else 0
        except Exception:
            soil_encoded = 0

        # Encode crop
        try:
            crop_encoded = le_crop.transform([crop_name])[0] if le_crop else 0
        except Exception:
            crop_encoded = 0

        # Compute NPK deficits vs crop optimum
        opt           = OPTIMAL_NPK.get(crop_name, (20, 20, 20))
        def_n         = max(0, opt[0] - N)
        def_p         = max(0, opt[1] - P)
        def_k         = max(0, opt[2] - K)
        total_deficit = def_n + def_p + def_k

        # Build 13-feature vector — must match FEATURE_COLS order in train_model.py
        features = np.array([[
            temp, humidity, moisture,
            soil_encoded, crop_encoded,
            N, K, P, pH,
            def_n, def_p, def_k, total_deficit
        ]])

        prediction = model.predict(features)
        fertilizer_name = le_fert.inverse_transform(prediction)[0] if le_fert else str(prediction[0])

        # Calculate recommended quantity with crop-specific adjustments
        fert_quantity = calculate_fertilizer_quantity(fertilizer_name, def_n, def_p, def_k, crop_name)

        # Get detailed application instructions
        instructions = get_application_instructions(fertilizer_name, fert_quantity, crop_name)

        # Save to DB with quantity
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            c = conn.cursor()
            c.execute('''INSERT INTO predictions
                         (user_id, nitrogen, phosphorus, potassium, ph, crop, soil, fertilizer_result, quantity_kg_per_ha)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                      (session['user_id'], N, P, K, pH, crop_name, soil_name, fertilizer_name, fert_quantity))
            conn.commit()
            conn.close()
        except Exception as db_error:
            print(f"Warning: Could not save prediction: {db_error}")

        return render_template(
            "result.html",
            fertilizer=fertilizer_name,
            crop=crop_name,
            soil=soil_name,
            temp=temp,
            humidity=humidity,
            weather=weather_desc,
            location=city,
            fertilizer_quantity=fert_quantity,
            nitrogen=N,
            phosphorus=P,
            potassium=K,
            ph_value=pH,
            optimal_n=opt[0],
            optimal_p=opt[1],
            optimal_k=opt[2],
            deficit_n=def_n,
            deficit_p=def_p,
            deficit_k=def_k,
            instructions=instructions
        )

    except Exception as e:
        print(f"[ERROR] Prediction error: {e}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", 500
    
@app.route("/ping")
def ping():
    return "alive"

# -------------------------------
# 7. RUN SERVER
# -------------------------------
if __name__ == "__main__":
    print("🚀 GreenGrow Server Running...")
    print("👉 http://127.0.0.1:5000")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)