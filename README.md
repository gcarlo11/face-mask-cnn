# Analisis dan Prediksi Serangan Jantung dengan K-Nearest Neighbor (K-NN)

## 📌 Pendahuluan

Proyek ini bertujuan untuk membangun model klasifikasi menggunakan algoritma **K-Nearest Neighbor (K-NN)** untuk memprediksi kemungkinan seorang pasien terkena serangan jantung. Dataset yang digunakan bersumber dari Kaggle dan berisi berbagai atribut medis serta demografis pasien. Penyakit jantung merupakan penyebab utama kematian secara global, sehingga deteksi dini yang akurat dapat meningkatkan hasil kesehatan pasien secara signifikan.

## 📂 Konten Repositori

  - `KNN_Heart_Attack (1).ipynb`: Notebook Jupyter yang berisi seluruh kode untuk analisis, pra-pemrosesan data, pemodelan, dan evaluasi.
  - `heart.csv`: File dataset yang digunakan (diperlukan untuk menjalankan notebook).

## 🛠️ Persyaratan

Untuk menjalankan kode ini, Anda memerlukan lingkungan Python dengan pustaka-pustaka berikut:

  - pandas
  - numpy
  - matplotlib
  - seaborn
  - scikit-learn

Anda dapat menginstalnya dengan perintah:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## 📊 Detail Dataset

Dataset ini memiliki 303 baris dan 14 kolom. Setiap baris mewakili data satu pasien, dan setiap kolom mewakili atribut medis. Target yang ingin diprediksi adalah kolom **`output`**, yang merupakan klasifikasi biner:

  - `1`: Pasien memiliki penyakit jantung
  - `0`: Pasien tidak memiliki penyakit jantung

## 📈 Alur Proyek

### 1\. Pra-pemrosesan Data

  - **Pengecekan Data Kosong:** Dilakukan pengecekan untuk memastikan tidak ada nilai yang hilang (`null`) dalam dataset. Hasilnya menunjukkan dataset bersih dari missing data.
  - **Pengecekan Keseimbangan Kelas:** Menggunakan `sns.countplot(y)` untuk memvisualisasikan distribusi kelas target (`output`). Grafik menunjukkan bahwa dataset **cukup seimbang**, yang ideal untuk klasifikasi.

### 2\. Normalisasi Data

Karena K-NN mengandalkan perhitungan jarak antar-fitur, normalisasi data sangat penting untuk memastikan semua fitur memiliki skala yang sama. Ini mencegah fitur dengan rentang nilai besar mendominasi perhitungan jarak. Untuk ini, `StandardScaler` digunakan untuk menskalakan fitur.

### 3\. Pembagian Data

Dataset dibagi menjadi data latih (**training data**) dan data uji (**testing data**) dengan rasio **80:20**.

  - **Data Latih (`X_train`, `y_train`)**: Digunakan untuk melatih model.
  - **Data Uji (`X_test`, `y_test`)**: Digunakan untuk menguji performa model yang telah dilatih.

### 4\. Pemodelan K-NN

Model `KNeighborsClassifier` diimpor dari pustaka `scikit-learn`.

### 5\. Menentukan Nilai K Terbaik

Performa model K-NN sangat bergantung pada nilai hyperparameter $k$. Dilakukan pengujian untuk menemukan nilai $k$ terbaik dari 1 hingga 10, dengan mengevaluasi metrik berikut pada data uji:

  - **Akurasi (Accuracy)**: Proporsi prediksi yang benar.
  - **Presisi (Precision)**: Proporsi kasus positif yang diprediksi dengan benar dari total prediksi positif.
  - **Recall**: Proporsi kasus positif yang sebenarnya berhasil ditemukan oleh model.

### 6\. Evaluasi Hasil dan Kesimpulan

Pengujian menemukan bahwa nilai $k$ terbaik adalah **6**, yang menghasilkan akurasi tertinggi sebesar **0.934**. Model ini kemudian dievaluasi menggunakan metrik lengkap, termasuk **F1-Score**, dan divisualisasikan dengan **Confusion Matrix**.

#### Hasil Evaluasi Model (k=6)

  - **Akurasi**: 0.93
  - **Presisi**: 0.97
  - **Recall**: 0.91
  - **F1 Score**: 0.94

#### Confusion Matrix

  - **True Negative (TN)**: 27
  - **False Positive (FP)**: 2
  - **False Negative (FN)**: 3
  - **True Positive (TP)**: 29

**Kesimpulan:** Model K-NN dengan nilai $k=6$ menunjukkan performa yang sangat baik dalam memprediksi penyakit jantung, dengan akurasi pengujian sebesar 93%. Akurasi data latih dan data uji yang tidak jauh berbeda menunjukkan bahwa model ini tidak mengalami **overfitting** dan dapat digambarkan sebagai **"Good Fit"**.
