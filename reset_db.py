import sqlite3

conn = sqlite3.connect("database.db")
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS recommendations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    category TEXT,
    image_path TEXT,
    score REAL
)
""")

conn.commit()
conn.close()
print("Tabel recommendations berhasil dibuat ulang")
