# evaluation/eval_human.py
import sqlite3

"""
Rekap Evaluasi Manusia (Human Evaluation)
"""

conn = sqlite3.connect("database.db")
cur = conn.cursor()

cur.execute("""
SELECT
    AVG(skor_kesesuaian),
    AVG(skor_estetika),
    COUNT(*)
FROM human_evaluation
""")

avg_kesesuaian, avg_estetika, total = cur.fetchone()
conn.close()

print("=== HASIL HUMAN EVALUATION ===")
print("Total Penilaian:", total)
print("Rata-rata Kesesuaian:", round(avg_kesesuaian or 0, 2))
print("Rata-rata Estetika:", round(avg_estetika or 0, 2))
