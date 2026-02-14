import sqlite3

DB_NAME = "database.db"

def create_admin():
    conn = sqlite3.connect(DB_NAME)
    cur = conn.cursor()

    username = input("Masukkan username admin: ")
    password = input("Masukkan password admin: ")

    # cek apakah user sudah ada
    cur.execute("SELECT id FROM users WHERE username=?", (username,))
    if cur.fetchone():
        cur.execute(
            "UPDATE users SET role='admin' WHERE username=?",
            (username,)
        )
        print("User sudah ada, role diubah menjadi ADMIN.")
    else:
        cur.execute(
            "INSERT INTO users (username, password, role) VALUES (?,?,?)",
            (username, password, "admin")
        )
        print("Admin baru berhasil dibuat.")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_admin()
