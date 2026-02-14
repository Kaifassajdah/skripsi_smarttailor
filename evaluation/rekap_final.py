import sqlite3
import pandas as pd

def hitung_rekap_final():
    conn = sqlite3.connect("database.db")
    
    # 1. Ambil Akurasi Model Terakhir (CNN & Top-5)
    # Kita asumsikan akurasi Top-5 diambil dari model_metrics atau hasil running eval_recommend.py
    cur = conn.cursor()
    cur.execute("SELECT accuracy FROM model_metrics ORDER BY id DESC LIMIT 1")
    res_cnn = cur.fetchone()
    accuracy_cnn = (res_cnn[0] * 100) if res_cnn else 0

    # 2. Ambil Rata-rata Skor Pakar (Human Evaluation)
    cur.execute("""
        SELECT 
            AVG(skor_kesesuaian) as avg_sesuai,
            AVG(skor_estetika) as avg_estetika,
            COUNT(*) as total_uji
        FROM human_evaluation
    """)
    avg_sesuai, avg_estetika, total_uji = cur.fetchone()
    
    # Konversi skala 1-5 ke persen agar bisa dibandingkan dengan CNN
    # Rumus: (Skor / 5) * 100
    avg_pakar_persen = (( (avg_sesuai or 0) + (avg_estetika or 0) ) / 10) * 100

    print("===============================================")
    print("📊 REKAPITULASI HASIL PENELITIAN SMARTTAILOR")
    print("===============================================")
    print(f"Total Data Diuji Pakar : {total_uji} Data")
    print(f"Akurasi Model CNN      : {accuracy_cnn:.2f}%")
    print(f"Rata-rata Skor Pakar   : {((avg_sesuai or 0) + (avg_estetika or 0))/2:.2f} / 5.00")
    print(f"Persentase Kelayakan   : {avg_pakar_persen:.2f}%")
    print("===============================================")

    # 3. Tampilkan Tabel Perbandingan Detail per Riwayat
    print("\n📋 TABEL PERBANDINGAN PER DATA (SAMPEL):")
    query_tabel = """
        SELECT 
            h.id, 
            h.tubuh, 
            r.category, 
            MAX(r.score) * 100 as cnn_score,
            ev.skor_kesesuaian,
            ev.skor_estetika
        FROM history h
        JOIN recommendations r ON h.id = r.history_id
        JOIN human_evaluation ev ON h.id = ev.history_id
        GROUP BY h.id
    """
    df = pd.read_sql_query(query_tabel, conn)
    print(df.to_string(index=False))
    
    conn.close()

if __name__ == "__main__":
    hitung_rekap_final()