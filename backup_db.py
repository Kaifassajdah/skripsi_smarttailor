import sqlite3
from datetime import datetime

DB_NAME = "database.db"

conn = sqlite3.connect(DB_NAME)
cur = conn.cursor()

# ================= USERS =================
cur.execute("""
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE,
    password TEXT,
    role TEXT DEFAULT 'user',
    last_login TEXT
)
""")

# ================= HISTORY =================
cur.execute("""
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    input_image TEXT,
    created_at TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
)
""")

# ================= RECOMMENDATIONS =================
cur.execute("""
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    history_id INTEGER,
    category TEXT,
    image_path TEXT,
    score REAL,
    FOREIGN KEY(history_id) REFERENCES history(id)
)
""")

# ================= MODEL METRICS =================
cur.execute("""
CREATE TABLE IF NOT EXISTS model_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    accuracy REAL,
    precision REAL,
    recall REAL,
    f1_score REAL,
    created_at TEXT
)
""")

# ================= DEFAULT METRICS (JIKA KOSONG) =================
cur.execute("SELECT COUNT(*) FROM model_metrics")
if cur.fetchone()[0] == 0:
    cur.execute("""
        INSERT INTO model_metrics
        (accuracy, precision, recall, f1_score, created_at)
        VALUES (?,?,?,?,?)
    """, (
        0.90, 0.88, 0.87, 0.87,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ))

conn.commit()
conn.close()

print("✅ Database berhasil di-inisialisasi / diperbarui")
