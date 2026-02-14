import os
import sqlite3
import json
import subprocess
import sys
import uuid
from datetime import datetime, timedelta
from flask import Flask, jsonify, send_from_directory, render_template, request, redirect, session, flash, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename
from similarity import recommend
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle
from reportlab.lib.units import inch
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import ListFlowable, ListItem

# ================= APP CONFIG =================
app = Flask(__name__)
app.secret_key = "smarttailor"
app.config["UPLOAD_FOLDER"] = "static/uploads"
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ================= DATABASE =================
def get_db():
    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    return conn

# ================= AUTH DECORATOR =================
def login_required():
    if "user_id" not in session:
        return False
    return True

# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = get_db()
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE username=?", (username,))
        user = cur.fetchone()

        if user:
            db_password = user["password"]

            # Kalau sudah hashed
            if db_password.startswith("scrypt") or db_password.startswith("pbkdf2"):
                valid = check_password_hash(db_password, password)
            else:
                valid = (db_password == password)
                if valid:
                    # Upgrade ke hash
                    hashed = generate_password_hash(password)
                    cur.execute(
                        "UPDATE users SET password=? WHERE id=?",
                        (hashed, user["id"])
                    )
                    conn.commit()

            if valid:
                session["user_id"] = user["id"]
                session["role"] = user["role"]  # 🔥 penting

                conn.close()

                # Redirect sesuai role
                if user["role"] == "admin":
                    return redirect(url_for("admin_dashboard"))
                else:
                    return redirect(url_for("beranda"))

        conn.close()
        flash("Username atau password salah!", "danger")

    return render_template("login.html")

# ================= REGISTER =================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        try:
            conn = get_db()
            cur = conn.cursor()
            cur.execute(
                "INSERT INTO users (username, password, role) VALUES (?,?,?)",
                (username, password, "user")
            )
            conn.commit()
            conn.close()

            flash("Registrasi berhasil. Silakan login.")
            return redirect("/login")

        except sqlite3.IntegrityError:
            flash("Username sudah digunakan!")
            return redirect("/register")

    return render_template("register.html")

# ================= LUPA PASSWORD =================
@app.route("/lupa_password", methods=["GET", "POST"])
def lupa_password():

    if request.method == "POST":
        username = request.form["username"]
        new_password = request.form["password"]

        conn = sqlite3.connect("database.db")
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()

        cur.execute("SELECT * FROM users WHERE username = ?", (username,))
        user = cur.fetchone()

        if not user:
            conn.close()
            flash("Username tidak ditemukan!", "danger")
            return redirect(url_for("lupa_password"))

        hashed_password = generate_password_hash(new_password)

        cur.execute("""
            UPDATE users 
            SET password=? 
            WHERE username=?
        """, (hashed_password, username))

        conn.commit()
        conn.close()

        flash("Password berhasil direset!", "success")
        return redirect(url_for("login"))

    return render_template("lupa_password.html")

# ================= RESET PASSWORD =================
@app.route("/reset_password/<token>", methods=["GET", "POST"])
def reset_password(token):

    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT * FROM users 
        WHERE reset_token=? 
    """, (token,))
    user = cur.fetchone()

    if not user:
        conn.close()
        return "Token tidak valid!"

    expiry = datetime.strptime(user["reset_token_expiry"], "%Y-%m-%d %H:%M:%S")

    if datetime.now() > expiry:
        conn.close()
        return "Token sudah kadaluarsa!"

    if request.method == "POST":
        new_password = request.form["password"]
        hashed_password = generate_password_hash(new_password)

        cur.execute("""
            UPDATE users 
            SET password=?, reset_token=NULL, reset_token_expiry=NULL
            WHERE id=?
        """, (hashed_password, user["id"]))

        conn.commit()
        conn.close()

        flash("Password berhasil direset!", "success")
        return redirect(url_for("login"))

    conn.close()
    return render_template("reset_password.html")

# ================= LOGOUT =================
@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")

# ================= BERANDA =================
@app.route("/")
def beranda():
    if not login_required():
        return redirect("/login")
    return render_template("beranda.html")

# ================= REKOMENDASI =================
@app.route("/rekomendasi", methods=["GET", "POST"])
def rekomendasi_page():
    if not login_required():
        return redirect("/login")

    tubuh = request.args.get("tubuh") or request.form.get("tubuh")

    if not tubuh:
        flash("Silakan pilih bentuk tubuh terlebih dahulu.")
        return redirect("/")

    input_image = None
    category = None
    confidence = None
    results = []

    if request.method == "POST":
        file = request.files["image"]
        filename = secure_filename(file.filename)

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(save_path)

        input_image = "/" + save_path.replace("\\", "/")

        category, confidence, results = recommend(save_path, tubuh)

        if confidence < 0.5:
            flash("Gambar kurang jelas, silakan upload ulang.")
            return redirect(f"/rekomendasi?tubuh={tubuh}")

    return render_template(
        "rekomendasi.html",
        tubuh=tubuh,
        input_image=input_image,
        category=category,
        confidence=confidence,
        results=results
    )

# ================= SIMPAN RIWAYAT =================
@app.route("/simpan", methods=["POST"])
def simpan_riwayat():
    if not login_required():
        return redirect("/login")

    conn = get_db()
    cur = conn.cursor()

    cur.execute(
        "INSERT INTO history (user_id, input_image, tubuh) VALUES (?,?,?)",
        (session["user_id"], request.form["input_image"], request.form["tubuh"])
    )

    history_id = cur.lastrowid

    for img, score in zip(
        request.form.getlist("image_path[]"),
        request.form.getlist("score[]")
    ):
        cur.execute(
            "INSERT INTO recommendations (history_id, category, image_path, score) VALUES (?,?,?,?)",
            (history_id, request.form["category"], img, float(score))
        )

    conn.commit()
    conn.close()

    return redirect("/riwayat")

# ================= RIWAYAT =================
@app.route("/riwayat")
def riwayat():
    if not login_required():
        return redirect("/login")

    conn = get_db()
    cur = conn.cursor()

    cur.execute("""
        SELECT h.id, h.input_image, h.tubuh,
               r.category, r.image_path, r.score
        FROM history h
        JOIN recommendations r ON h.id = r.history_id
        WHERE h.user_id=?
        ORDER BY h.id DESC
    """, (session["user_id"],))

    rows = cur.fetchall()
    conn.close()

    history = {}
    for hid, input_img, tubuh, cat, img, sc in rows:
        if hid not in history:
            history[hid] = {
                "input_image": input_img,
                "category": cat,
                "tubuh": tubuh,
                "results": []
            }
        history[hid]["results"].append({
            "image": img,
            "score": sc
        })

    return render_template("riwayat.html", history=history)

# ================= HAPUS RIWAYAT =================
@app.route("/hapus_riwayat", methods=["POST"])
def hapus_riwayat():
    if "user_id" not in session:
        return redirect("/login")

    ids = request.form.getlist("hapus_id[]")
    if not ids:
        return redirect("/riwayat")

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    for hid in ids:
        # hapus rekomendasi dulu (child)
        cur.execute("DELETE FROM recommendations WHERE history_id=?", (hid,))
        # lalu hapus history (parent)
        cur.execute("DELETE FROM history WHERE id=? AND user_id=?", (hid, session["user_id"]))

    conn.commit()
    conn.close()

    return redirect("/riwayat")

# ================= ADMIN DASHBOARD =================
@app.route("/admin")
def admin_dashboard():
    if session.get("role") != "admin":
        return redirect("/login")

    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # 1. Ambil Metrik Sistem
    cur.execute("SELECT * FROM model_metrics ORDER BY id DESC LIMIT 1")
    system_metrics = cur.fetchone()

    # 2. Ambil Metrik Pakar (Statistik Manual)
    cur.execute("SELECT * FROM expert_metrics ORDER BY id DESC LIMIT 1")
    expert_metrics = cur.fetchone()

    # 3. Ambil rata-rata rating dari Human Evaluation (Tabel ratings)
    cur.execute("""
        SELECT 
            AVG(skor_kesesuaian),
            AVG(skor_estetika)
        FROM ratings
    """)

    avg_rating = cur.fetchone()

    avg_kesesuaian = round(avg_rating[0], 2) if avg_rating[0] else 0
    avg_estetika = round(avg_rating[1], 2) if avg_rating[1] else 0

    conn.close()

    confusion_matrix = None
    if system_metrics and system_metrics["confusion_matrix"]:
        confusion_matrix = json.loads(system_metrics["confusion_matrix"])

    return render_template(
        "admin.html",
        system_metrics=system_metrics,
        expert_metrics=expert_metrics,
        confusion_matrix=confusion_matrix,
        avg_kesesuaian=avg_kesesuaian,
        avg_estetika=avg_estetika
    )

# ================= ADMIN USERS =================
@app.route("/admin/users")
def admin_users():
    if session.get("role") != "admin":
        return redirect("/login")

    keyword = request.args.get("q", "")

    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    if keyword:
        cur.execute("""
            SELECT id, username, last_login
            FROM users
            WHERE username LIKE ?
            ORDER BY id DESC
        """, ('%' + keyword + '%',))
    else:
        cur.execute("""
            SELECT id, username, last_login
            FROM users
            ORDER BY id DESC
        """)

    users = cur.fetchall()
    conn.close()

    return render_template(
        "admin_users.html",
        users=users,
        keyword=keyword
    )

# ================= ADMIN RIWAYAT =================
@app.route("/admin/riwayat")
def admin_riwayat():

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Ambil semua riwayat + join ke tabel users
    cur.execute("""
        SELECT h.id, h.user_id, h.input_image, h.tubuh, u.username
        FROM history h
        JOIN users u ON h.user_id = u.id
        ORDER BY h.id DESC
    """)

    histories = cur.fetchall()

    data = []

    for h in histories:

        # Ambil rekomendasi untuk tiap history
        cur.execute("""
            SELECT image_path, score
            FROM recommendations
            WHERE history_id = ?
        """, (h["id"],))

        recs = cur.fetchall()

        cur.execute("SELECT * FROM ratings WHERE history_id = ?", (h["id"],))
        rating = cur.fetchone()

        status = "Sudah Dinilai" if rating else "Belum Dinilai"

        data.append({
            "id": h["id"],
            "username": h["username"],
            "input_image": h["input_image"],
            "tubuh": h["tubuh"],
            "results": recs,
            "status": status
        })

    conn.close()

    return render_template("admin_riwayat.html", data=data)

# ================= ADMIN RATING =================
@app.route("/admin/rating/<int:history_id>", methods=["GET", "POST"])
def admin_rating(history_id):

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Ambil data history
    cur.execute("SELECT * FROM history WHERE id = ?", (history_id,))
    history = cur.fetchone()

    # Ambil rekomendasi
    cur.execute("""
        SELECT * FROM recommendations
        WHERE history_id = ?
    """, (history_id,))
    recommendations = cur.fetchall()

    if request.method == "POST":

        skor_kesesuaian = request.form["skor_kesesuaian"]
        skor_estetika = request.form["skor_estetika"]
        komentar = request.form["komentar"]

        cur.execute("""
            INSERT INTO ratings
            (history_id, skor_kesesuaian, skor_estetika, komentar, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (
            history_id,
            skor_kesesuaian,
            skor_estetika,
            komentar,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))

        conn.commit()
        conn.close()

        return redirect("/admin/riwayat")

    conn.close()

    return render_template(
        "admin_rating.html",
        history=history,
        recommendations=recommendations
    )

# ================= ADMIN RATING =================
@app.route("/admin/evaluasi_model")
def evaluasi_model():

    conn = sqlite3.connect("database.db")
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("SELECT * FROM model_metrics ORDER BY id DESC LIMIT 1")
    metrics = cur.fetchone()

    cur.execute("SELECT accuracy, f1_score, created_at FROM model_metrics ORDER BY id DESC")
    history_metrics = cur.fetchall()

    confusion_matrix = None
    classification_report = None

    if metrics:
        import json
        if metrics["confusion_matrix"]:
            confusion_matrix = json.loads(metrics["confusion_matrix"])
        if metrics["classification_report"]:
            classification_report = json.loads(metrics["classification_report"])

    conn.close()

    return render_template(
        "admin_evaluasi_model.html",
        metrics=metrics,
        confusion_matrix=confusion_matrix,
        classification_report=classification_report,
        history_metrics=history_metrics
    )

# ================= ADMIN EXPORT EVALUASI PDF =================
@app.route("/admin/export_evaluasi_pdf")
def export_evaluasi_pdf():

    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = sqlite3.connect("database.db")
    cur = conn.cursor()

    cur.execute("""
        SELECT accuracy, precision, recall, f1_score, created_at
        FROM model_metrics
        ORDER BY id DESC LIMIT 1
    """)
    row = cur.fetchone()
    conn.close()

    file_path = "evaluasi_model.pdf"
    doc = SimpleDocTemplate(file_path)
    elements = []

    styles = getSampleStyleSheet()
    elements.append(Paragraph("<b>Laporan Evaluasi Model CNN - SmartTailor</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    data = [
        ["Metric", "Value"],
        ["Accuracy", f"{row[0]*100:.2f}%"],
        ["Precision", f"{row[1]*100:.2f}%"],
        ["Recall", f"{row[2]*100:.2f}%"],
        ["F1-Score", f"{row[3]*100:.2f}%"],
        ["Tanggal Evaluasi", row[4]]
    ]

    table = Table(data, colWidths=[2*inch, 2*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),colors.grey),
        ('GRID',(0,0),(-1,-1),1,colors.black),
        ('ALIGN',(1,1),(-1,-1),'CENTER')
    ]))

    elements.append(table)
    doc.build(elements)

    return send_file(file_path, as_attachment=True)

# ================= ADMIN EXPORT EVALUASI PDF =================
@app.route("/admin/run_evaluasi_model", methods=["POST"])
def run_evaluasi_model():

    if "user_id" not in session:
        return redirect(url_for("login"))

    try:
        # Jalankan evaluate_model.py
        result = subprocess.run(
            [sys.executable, "evaluate_model.py"],
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            flash("Evaluasi model berhasil dijalankan!", "success")
        else:
            flash("Terjadi error saat evaluasi model.", "danger")
            print(result.stderr)

    except Exception as e:
        flash(f"Error: {str(e)}", "danger")

    return redirect(url_for("evaluasi_model"))

# ================= API REKOMENDASI =================
@app.route("/api/rekomendasi", methods=["POST"])
def api_rekomendasi():
    if "user_id" not in session:
        return jsonify({"error": "Unauthorized"}), 401

    tubuh = request.form.get("tubuh")

    if not tubuh:
        return jsonify({"error": "Tubuh tidak valid"}), 400

    file = request.files["image"]
    filename = secure_filename(file.filename)

    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(save_path)

    category, confidence, results = recommend(save_path, tubuh)

    return jsonify({
        "input_image": "/" + save_path.replace("\\", "/"),
        "category": category,
        "confidence": confidence,
        "results": results
    })

@app.route('/dataset/<path:filename>')
def serve_dataset(filename):
    return send_from_directory('dataset', filename)

# ================= RUN =================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
