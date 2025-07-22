
# 📘 K-Nearest Neighbors (KNN) — Penjelasan Lengkap + Contoh Python

KNN adalah salah satu algoritma **Machine Learning** paling sederhana dan intuitif. Ia termasuk **supervised learning** dan sering digunakan untuk **klasifikasi** dan **regresi**.

---

## 🧠 Konsep Dasar

> KNN memprediksi data baru dengan **melihat K tetangga terdekat** dari data latih.  
> Mirip seperti kamu mencari teman yang mirip untuk menilai kepribadian seseorang.

---

## 📍 Cara Kerja KNN

1. Simpan semua data latih.
2. Saat ada data baru, **hitung jarak** ke semua data latih.
3. Pilih **K tetangga terdekat** (berdasarkan jarak terkecil).
4. Lakukan **voting**:
   - Untuk klasifikasi → mayoritas label
   - Untuk regresi → rata-rata nilai

---

## 👥 Analogi Nyata

Bayangkan kamu anak baru dan ingin menebak apakah seorang siswa "Rajin" atau "Santai".  
Kamu menilai berdasarkan **nilai ulangan** dan apakah dia **sering telat**.

Kamu lihat siswa lain:
| Nilai | Telat (0/1) | Label   |
|-------|-------------|---------|
| 90    | 0           | Rajin   |
| 85    | 0           | Rajin   |
| 60    | 1           | Santai  |
| 65    | 1           | Santai  |
| 80    | 0           | Rajin   |

Siswa baru: **(70, 1)**  
Lalu kamu hitung siapa yang paling mirip, dan prediksi dia Rajin atau Santai.

---

## 🧮 Perhitungan Manual (Euclidean Distance)

\[
\text{Jarak} = \sqrt{(x_1 - y_1)^2 + (x_2 - y_2)^2}
\]

Contoh hitung ke (65, 1):
```plaintext
√((70 - 65)^2 + (1 - 1)^2) = √25 = 5
```

Lakukan ke semua data, ambil 3 yang paling dekat, voting label-nya → **Prediksi: Santai**

---

## 💻 Kode Python Sederhana (Manual, tanpa sklearn)

```python
import math

# Data latih
data_latih = [
    (90, 0, "Rajin"),
    (85, 0, "Rajin"),
    (60, 1, "Santai"),
    (65, 1, "Santai"),
    (80, 0, "Rajin")
]

# Data yang ingin diprediksi
data_baru = (70, 1)

# Fungsi untuk menghitung jarak Euclidean
def hitung_jarak(data1, data2):
    return math.sqrt((data1[0] - data2[0])**2 + (data1[1] - data2[1])**2)

# Hitung semua jarak dan simpan labelnya
jarak_dan_label = []
for nilai, telat, label in data_latih:
    jarak = hitung_jarak((nilai, telat), data_baru)
    jarak_dan_label.append((jarak, label))

# Urutkan dan ambil 3 tetangga terdekat
jarak_dan_label.sort()
K = 3
tetangga_terdekat = jarak_dan_label[:K]

# Voting
label_count = {}
for _, label in tetangga_terdekat:
    label_count[label] = label_count.get(label, 0) + 1

# Ambil label terbanyak
prediksi = max(label_count, key=label_count.get)

# Output
print("Prediksi untuk data baru:", data_baru, "adalah:", prediksi)
```

---

## 📏 Jenis-Jenis Jarak yang Bisa Dipakai

| Nama               | Cocok Untuk               | Rumus Singkat                      |
|--------------------|---------------------------|------------------------------------|
| Euclidean          | Angka kontinu              | √((x₁ - y₁)² + (x₂ - y₂)² + ...)   |
| Manhattan          | Grid / skala beda          | |x₁ - y₁| + |x₂ - y₂| + ...         |
| Minkowski (p=n)    | Umum, kombinasi            | ((∑|x - y|^p)^(1/p))                |
| Cosine Similarity  | Teks / arah vektor         | 1 - (A · B) / (|A| × |B|)           |
| Hamming Distance   | Data biner / kategori      | Hitung posisi yang berbeda         |

---

## 🛠️ Kapan KNN Digunakan?

- Deteksi penyakit (klasifikasi: sakit atau tidak)
- Rekomendasi film / produk
- Prediksi nilai atau kategori siswa
- Deteksi spam email
- Pengenalan wajah

---

## ⚠️ Kelebihan & Kekurangan

### ✅ Kelebihan:
- Sederhana dan mudah dipahami
- Tidak perlu pelatihan model (lazy learning)
- Cocok untuk data kecil dan jelas

### ❌ Kekurangan:
- Lambat kalau data banyak (harus hitung semua jarak)
- Sensitif terhadap fitur yang beda skala
- Butuh pemilihan nilai **K** yang tepat

---

## 📚 Kesimpulan

KNN sangat cocok untuk pemula yang ingin belajar machine learning secara intuitif.  
Dengan konsep “mirip dengan siapa”, kamu bisa memprediksi hasil dari data baru hanya dengan membandingkan!

---

## 📎 Referensi
- https://scikit-learn.org/stable/modules/neighbors.html
- https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm

---

## ✨ Next Step
👉 Coba ubah `K`, ganti data, atau terapkan ke kasus kamu sendiri!
