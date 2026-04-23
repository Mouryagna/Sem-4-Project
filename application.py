import os
import sys
import json
import sqlite3
import urllib.request
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import pandas as pd
import numpy as np
from datetime import datetime
from functools import wraps
from werkzeug.security import check_password_hash, generate_password_hash

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATABASE_PATH = os.path.join(PROJECT_ROOT, 'users.db')

app = Flask(__name__, template_folder='template', static_folder='static')
app.secret_key = 'delhi-aqi-vayusense-secret-2026'


def get_db_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_auth_db():
    conn = get_db_connection()
    try:
        conn.execute(
            '''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                email TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            '''
        )
        conn.commit()
    finally:
        conn.close()


def get_user_by_username(username):
    conn = get_db_connection()
    try:
        return conn.execute(
            'SELECT id, username, email, password FROM users WHERE username = ?',
            (username,)
        ).fetchone()
    finally:
        conn.close()


def get_user_by_email(email):
    conn = get_db_connection()
    try:
        return conn.execute(
            'SELECT id, username, email, password FROM users WHERE email = ?',
            (email,)
        ).fetchone()
    finally:
        conn.close()


def create_user(username, email, password):
    conn = get_db_connection()
    try:
        conn.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username, email, generate_password_hash(password))
        )
        conn.commit()
    finally:
        conn.close()


def verify_user_password(user_row, password):
    stored_password = user_row['password']
    if not stored_password:
        return False

    # Support legacy plaintext rows and migrate them on first successful login.
    if stored_password.startswith('pbkdf2:') or stored_password.startswith('scrypt:'):
        return check_password_hash(stored_password, password)

    if stored_password == password:
        conn = get_db_connection()
        try:
            conn.execute(
                'UPDATE users SET password = ? WHERE id = ?',
                (generate_password_hash(password), user_row['id'])
            )
            conn.commit()
        finally:
            conn.close()
        return True

    return False


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login', next=request.path))
        return view_func(*args, **kwargs)
    return wrapped_view


def get_aqi_category(aqi):
    if aqi <= 50:
        return 'Good', '#10b981', 'fas fa-smile-beam'
    elif aqi <= 100:
        return 'Satisfactory', '#22c55e', 'fas fa-smile'
    elif aqi <= 200:
        return 'Moderate', '#f59e0b', 'fas fa-meh'
    elif aqi <= 300:
        return 'Poor', '#f97316', 'fas fa-frown'
    elif aqi <= 400:
        return 'Very Poor', '#ef4444', 'fas fa-sad-tear'
    else:
        return 'Severe', '#dc2626', 'fas fa-skull-crossbones'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/dashboard')
@login_required
def dashboard():
    try:
        raw_path = os.path.join(PROJECT_ROOT, 'artifacts', 'raw.csv')
        df = pd.read_csv(raw_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df = df.sort_values('datetime').reset_index(drop=True)

        recent = df.tail(168)
        chart_labels = recent['datetime'].dt.strftime('%d %b %H:%M').tolist()
        chart_aqi = [round(v, 1) if not pd.isna(v) else 0 for v in recent['aqi'].tolist()]
        chart_pm25 = [round(v, 1) if not pd.isna(v) else 0 for v in recent['pm25'].tolist()]
        chart_pm10 = [round(v, 1) if not pd.isna(v) else 0 for v in recent['pm10'].tolist()]

        latest = df.iloc[-1]
        current_aqi = round(float(latest['aqi']), 1)
        cat, color, icon = get_aqi_category(current_aqi)

        stats = {
            'aqi': current_aqi, 'category': cat, 'color': color, 'icon': icon,
            'pm25': round(float(latest.get('pm25', 0)), 1),
            'pm10': round(float(latest.get('pm10', 0)), 1),
            'no2': round(float(latest.get('no2', 0)), 1),
            'so2': round(float(latest.get('so2', 0)), 1),
            'co': round(float(latest.get('co', 0)), 2),
            'o3': round(float(latest.get('o3', 0)), 1),
            'temperature': round(float(latest.get('temperature', 25)), 1),
            'humidity': round(float(latest.get('humidity', 60)), 1),
            'wind_speed': round(float(latest.get('wind_speed', 8)), 1),
            'datetime': latest['datetime'].strftime('%d %b %Y, %I:%M %p')
        }

        return render_template('dashboard.html',
                               chart_labels=json.dumps(chart_labels),
                               chart_aqi=json.dumps(chart_aqi),
                               chart_pm25=json.dumps(chart_pm25),
                               chart_pm10=json.dumps(chart_pm10),
                               stats=stats)
    except Exception as e:
        logging.error(f"Dashboard error: {e}")
        return render_template('dashboard.html', chart_labels='[]',
                               chart_aqi='[]', chart_pm25='[]', chart_pm10='[]',
                               stats=None, error=str(e))


@app.route('/predict')
@login_required
def predict():
    return render_template('predict.html')


@app.route('/realtime')
@login_required
def realtime():
    return render_template('realtime.html')


DELHI_LOCATIONS = [
    {'name': 'New Delhi',     'lat': 28.6139, 'lon': 77.2090},
    {'name': 'Anand Vihar',   'lat': 28.6469, 'lon': 77.3164},
    {'name': 'Noida',         'lat': 28.5355, 'lon': 77.3910},
    {'name': 'Gurgaon',       'lat': 28.4595, 'lon': 77.0266},
    {'name': 'Ghaziabad',     'lat': 28.6692, 'lon': 77.4538},
    {'name': 'Dwarka',        'lat': 28.5921, 'lon': 77.0460},
]

AQI_CATEGORIES = [
    (50, 'Good', '#10b981'),
    (100, 'Satisfactory', '#22c55e'),
    (200, 'Moderate', '#f59e0b'),
    (300, 'Poor', '#f97316'),
    (400, 'Very Poor', '#ef4444'),
    (500, 'Severe', '#dc2626'),
]


@app.route('/api/realtime')
@login_required
def api_realtime():
    """Fetch live AQI from Open-Meteo Air Quality API — free, no key needed."""
    stations = []
    for loc in DELHI_LOCATIONS:
        try:
            url = (
                f"https://air-quality-api.open-meteo.com/v1/air-quality"
                f"?latitude={loc['lat']}&longitude={loc['lon']}"
                f"&current=pm10,pm2_5,carbon_monoxide,nitrogen_dioxide,sulphur_dioxide,ozone,us_aqi"
                f"&timezone=Asia%2FKolkata"
            )
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())

            c = data['current']
            aqi = c.get('us_aqi', 0)
            cat, color = 'Severe', '#dc2626'
            for threshold, name, clr in AQI_CATEGORIES:
                if aqi <= threshold:
                    cat, color = name, clr
                    break

            stations.append({
                'name': loc['name'],
                'aqi': aqi,
                'category': cat,
                'color': color,
                'pm25': round(c.get('pm2_5', 0), 1),
                'pm10': round(c.get('pm10', 0), 1),
                'no2': round(c.get('nitrogen_dioxide', 0), 1),
                'so2': round(c.get('sulphur_dioxide', 0), 1),
                'co': round(c.get('carbon_monoxide', 0) / 1000, 2),
                'o3': round(c.get('ozone', 0), 1),
                'time': c.get('time', ''),
            })
        except Exception as e:
            logging.error(f"API error {loc['name']}: {e}")
    return jsonify({'success': len(stations) > 0, 'stations': stations})


@app.route('/info')
def info():
    return render_template('info.html')


@app.route('/api/predict', methods=['POST'])
@login_required
def api_predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Invalid JSON payload'}), 400

        required_fields = ['pm25', 'pm10', 'no2', 'so2', 'co', 'o3', 'hour', 'day', 'month', 'weekday', 'season']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            return jsonify({'success': False, 'error': f"Missing fields: {', '.join(missing_fields)}"}), 400

        custom_data = CustomData(
            pm25=float(data['pm25']), pm10=float(data['pm10']),
            no2=float(data['no2']), so2=float(data['so2']),
            co=float(data['co']), o3=float(data['o3']),
            hour=int(data['hour']), day=int(data['day']),
            month=int(data['month']), weekday=int(data['weekday']),
            season=data['season']
        )
        pred_df = custom_data.get_data_as_data_frame()
        pipeline = PredictPipeline()
        predicted_aqi = max(0.0, pipeline.predict(pred_df))
        cat, color, icon = get_aqi_category(predicted_aqi)

        if 'history' not in session:
            session['history'] = []

        record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'pm25': float(data['pm25']), 'pm10': float(data['pm10']),
            'no2': float(data['no2']), 'so2': float(data['so2']),
            'co': float(data['co']), 'o3': float(data['o3']),
            'hour': int(data['hour']), 'day': int(data['day']),
            'month': int(data['month']), 'season': data['season'],
            'predicted_aqi': round(predicted_aqi, 2), 'category': cat
        }
        session['history'] = session.get('history', []) + [record]
        session.modified = True

        return jsonify({'success': True, 'aqi': round(predicted_aqi, 2),
                        'category': cat, 'color': color, 'icon': icon})
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/history')
@login_required
def history():
    predictions = session.get('history', [])
    return render_template('history.html', predictions=predictions)


@app.route('/api/clear-history', methods=['POST'])
@login_required
def clear_history():
    session['history'] = []
    session.modified = True
    return jsonify({'success': True})


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        next_url = request.form.get('next') or request.args.get('next') or url_for('dashboard')

        if not username or not password:
            return render_template('login.html', error='Please fill in all fields', next_url=next_url)

        user = get_user_by_username(username)
        if not user or not verify_user_password(user, password):
            return render_template('login.html', error='Invalid username or password', next_url=next_url)

        session.clear()
        session['logged_in'] = True
        session['user_id'] = user['id']
        session['username'] = user['username']
        session['email'] = user['email']
        return redirect(next_url)

    next_url = request.args.get('next', url_for('dashboard'))
    return render_template('login.html', next_url=next_url)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if session.get('logged_in'):
        return redirect(url_for('dashboard'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirm_password = request.form.get('confirm_password', '')

        if not username or not email or not password or not confirm_password:
            return render_template('register.html', error='Please fill in all fields')

        if password != confirm_password:
            return render_template('register.html', error='Passwords do not match')

        if len(password) < 8:
            return render_template('register.html', error='Password must be at least 8 characters long')

        if get_user_by_username(username):
            return render_template('register.html', error='Username is already taken')

        if get_user_by_email(email):
            return render_template('register.html', error='Email is already registered')

        create_user(username, email, password)
        return render_template('login.html', success='Account created. You can sign in now.', next_url=url_for('dashboard'))

    return render_template('register.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))


init_auth_db()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
