import hashlib
import sqlite3

def get_connection():
    connection = sqlite3.connect('prognos.db')
    connection.row_factory = sqlite3.Row
    return connection

def init_db():
    connection = get_connection()
    cursor = connection.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        forecast_type TEXT NOT NULL,
        result TEXT NOT NULL,
        date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    connection.commit()
    connection.close()

def regist_db(username, email, password_hash):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute('INSERT INTO users (username, email, password_hash) VALUES (?,?,?)', 
                   (username, email, password_hash))
    connection.commit()
    connection.close()

def add_prediction_to_history(user_id, forecast_type, result):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute('''
        INSERT INTO predictions (user_id, forecast_type, result) 
        VALUES (?, ?, ?)
    ''', (user_id, forecast_type, result))
    connection.commit()
    connection.close()

def create_forecast(user_id, forecast_type, result):
    add_prediction_to_history(user_id, forecast_type, result)
    
def login_db(username, password):
    connection = get_connection()
    cursor = connection.cursor()
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    cursor.execute('SELECT * FROM users WHERE username = ? AND password_hash = ?', 
                   (username, password_hash))
    user = cursor.fetchone()
    connection.close()
    return user

def check_user_exists(username, email):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM users WHERE username = ? OR email = ?', (username, email))
    user = cursor.fetchone()
    connection.close()
    return user

def get_user_predictions(user_id):
    connection = get_connection()
    cursor = connection.cursor()
    cursor.execute('SELECT * FROM predictions WHERE user_id = ? ORDER BY date DESC', (user_id,))
    predictions = cursor.fetchall()
    connection.close()
    return predictions