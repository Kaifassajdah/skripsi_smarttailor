import sqlite3
from datetime import datetime

DB_NAME = "database.db"

def migrate_human_evaluation():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    print("[INFO] Mengecek tabel human_evaluation ...")

    # ===============================
    # BUAT TABEL JIKA BELUM ADA
    # ===============================
    cur.execute("""
    CREATE TABLE IF NOT EXISTS human_evaluation (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        history_id INTEGER NOT NULL,
        admin_id INTEGER NOT NULL,
        skor_kesesuaian INTEGER CHECK(skor_kesesuaian BETWEEN 1 AND 5),
        skor_estetika INTEGER CHECK(skor_estetika BETWEEN 1 AND 5),
        komentar TEXT,
        created_at TEXT,
        FOREIGN KEY(history_id) REFERENCES history(id),
        FOREIGN KEY(admin_id) REFERENCES users(id)
    )
    """)

    conn.commit()

    cur.execute("PRAGMA table_info(history)")
    cols = [c[1] for c in cur.fetchall()]

    if "tubuh" not in cols:
        cur.execute("ALTER TABLE history ADD COLUMN tubuh TEXT")
        print("✅ Kolom tubuh ditambahkan ke history")
    else:
        print("ℹ️ Kolom tubuh sudah ada")
    # ===============================
    # CEK STRUKTUR TABEL
    # ===============================
    print("[INFO] Struktur tabel human_evaluation:")
    cur.execute("PRAGMA table_info(human_evaluation);")
    columns = cur.fetchall()
    for col in columns:
        print(col)

    # ===============================
    # OPTIONAL: INSERT DATA DUMMY (AMAN)
    # hanya jika tabel masih kosong
    # ===============================
    cur.execute("SELECT COUNT(*) FROM human_evaluation")
    count = cur.fetchone()[0]

    if count == 0:
        print("[INFO] Tabel masih kosong, insert contoh data (opsional)")

        cur.execute("""
        INSERT INTO human_evaluation
        (history_id, admin_id, skor_kesesuaian, skor_estetika, komentar, created_at)
        VALUES (?,?,?,?,?,?)
        """, (
            1,              # history_id (asumsi ada)
            1,              # admin_id (asumsi admin id = 1)
            4,              # skor_kesesuaian
            4,              # skor_estetika
            "Contoh penilaian awal",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        print("[SUCCESS] Data contoh berhasil ditambahkan")
    else:
        print("[INFO] Data sudah ada, tidak menambahkan data dummy")

    conn.close()
    print("[DONE] Migrasi human_evaluation selesai dengan aman")

if __name__ == "__main__":
    migrate_human_evaluation()
