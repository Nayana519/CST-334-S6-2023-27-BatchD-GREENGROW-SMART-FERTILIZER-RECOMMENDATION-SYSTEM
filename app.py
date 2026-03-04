from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import sqlite3
import joblib
import requests
import os
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
        c.execute("PRAGMA table_info(users)")
        columns = [col[1] for col in c.fetchall()]
        if 'avatar_filename' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN avatar_filename TEXT")
            conn.commit()
            print("✅ Database migration: Added avatar_filename column")
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
model       = joblib.load("model.pkl")              if os.path.exists("model.pkl")              else None
le_soil     = joblib.load("soil_encoder.pkl")       if os.path.exists("soil_encoder.pkl")       else None
le_crop     = joblib.load("crop_encoder.pkl")       if os.path.exists("crop_encoder.pkl")       else None
le_fert     = joblib.load("fertilizer_encoder.pkl") if os.path.exists("fertilizer_encoder.pkl") else None
OPTIMAL_NPK = joblib.load("optimal_npk.pkl")        if os.path.exists("optimal_npk.pkl")        else {}

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_KEY", "f94b34629dd45e28b659165a63fc7595")

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
        if data.get("cod") != 200:
            return 25.0, 50.0, "Weather Unavailable"
        temp = round(data["main"]["temp"], 1)
        humidity = data["main"]["humidity"]
        description = data["weather"][0]["description"].title()
        return temp, humidity, description
    except Exception:
        return 25.0, 50.0, "Weather Unavailable"

# -------------------------------
# 4. ROUTES
# -------------------------------
@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name     = request.form.get("name", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm  = request.form.get("confirm", "").strip()

        if not all([name, email, password]) or password != confirm:
            return render_template("register.html", error="Invalid input or passwords don't match")

        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            c = conn.cursor()
            hashed_pw = generate_password_hash(password)
            c.execute('INSERT INTO users (name, email, password) VALUES (?, ?, ?)',
                      (name, email, hashed_pw))
            conn.commit()
            conn.close()
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            return render_template("register.html", error="Email already exists")
        except Exception as e:
            print(f"Database error during registration: {e}")
            return render_template("register.html", error="Database error. Please try again later.")

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
    return redirect(url_for('landing'))

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
    return render_template("profile.html", user=user)

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
            except:
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
        c.execute('SELECT * FROM predictions WHERE user_id=? ORDER BY created_at DESC LIMIT 10',
                  (session['user_id'],))
        predictions = c.fetchall()
        conn.close()
        return render_template("history.html", predictions=predictions)
    except Exception as e:
        print(f"Error fetching history: {e}")
        return render_template("history.html", predictions=[])

# -------------------------------
# 5. PREDICTION ROUTE
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

        # Compute NPK deficits vs crop optimum (matches training feature engineering)
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

        # Save to DB
        try:
            conn = sqlite3.connect(DB_PATH, timeout=10)
            c = conn.cursor()
            c.execute('''INSERT INTO predictions
                         (user_id, nitrogen, phosphorus, potassium, ph, crop, soil, fertilizer_result)
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                      (session['user_id'], N, P, K, pH, crop_name, soil_name, fertilizer_name))
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
            weather=weather_desc
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", 500

# -------------------------------
# 6. RUN SERVER
# -------------------------------
if __name__ == "__main__":
    print("🚀 GreenGrow Server Running...")
    print("👉 http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)