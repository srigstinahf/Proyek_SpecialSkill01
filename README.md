# Prediksi Penjualan dari Anggaran Iklan Menggunakan Machine Learning Linear Regression

## Gambaran Umum Proyek

Proyek ini bertujuan untuk membangun model *machine learning* yang dapat memprediksi jumlah penjualan (*sales*) berdasarkan anggaran iklan yang dialokasikan pada tiga media, yaitu TV, Radio, dan Koran (*Newspaper*).

Tujuan utama proyek ini adalah mengetahui seberapa besar anggaran iklan memengaruhi penjualan, serta mengeksplorasi bagaimana penerapan *machine learning* dapat membantu perusahaan memperkirakan penjualan sebelum biaya iklan benar-benar dikeluarkan.

Dalam proyek ini, dilakukan pembangunan dan evaluasi satu algoritma *machine learning*, yaitu Linear Regression.

## Tujuan Proyek

- Menganalisis hubungan antara anggaran iklan di setiap media dan penjualan.
- Memeriksa kualitas data (*missing value* dan data duplikat) serta melakukan analisis eksploratif (*Exploratory Data Analysis*).
- Membangun model *machine learning* untuk memprediksi penjualan.
- Mengevaluasi performa model menggunakan metrik evaluasi regresi.
- Mengidentifikasi media iklan yang paling berpengaruh berdasarkan koefisien model.
- Menghasilkan *business insight* dan rekomendasi berdasarkan hasil analisis.

## Dataset

Dataset yang digunakan dalam proyek ini adalah **Advertising Dataset** yang dimuat langsung dari repositori GitHub.

- **Sumber:** [Advertising Dataset](https://raw.githubusercontent.com/selva86/datasets/master/Advertising.csv)
- **Jumlah Data:** 200 baris
- **Jumlah Kolom:** 4 kolom
- **Variabel Target:** `sales`

Deskripsi variabel:

| Kolom | Keterangan | Satuan |
|---|---|---|
| `TV` | Anggaran iklan di TV | Ribu dolar ($000) |
| `radio` | Anggaran iklan di Radio | Ribu dolar ($000) |
| `newspaper` | Anggaran iklan di Koran | Ribu dolar ($000) |
| `sales` | Hasil penjualan (variabel target) | Ribu unit |

Ringkasan statistik:

| Statistik | TV | radio | newspaper | sales |
|---|---:|---:|---:|---:|
| Rata-rata | 147,04 | 23,26 | 30,55 | 14,02 |
| Standar deviasi | 85,85 | 14,85 | 21,78 | 5,22 |
| Minimum | 0,70 | 0,00 | 0,30 | 1,60 |
| Maksimum | 296,40 | 49,60 | 114,00 | 27,00 |

Hasil pengecekan kualitas data menunjukkan tidak terdapat nilai kosong (*missing value*) maupun baris duplikat.

Dataset dimuat secara otomatis dari URL menggunakan `pandas.read_csv`.

## Tools dan Teknologi

Tools dan library yang digunakan dalam proyek ini:

- **Python:** Bahasa pemrograman utama.
- **Pandas dan NumPy:** Pengolahan dan manipulasi data.
- **Matplotlib dan Seaborn:** Visualisasi data.
- **Scikit-learn:** Pembagian data, pembangunan model, dan evaluasi machine learning.
- **Jupyter Notebook:** Lingkungan untuk analisis dan eksperimen.

## Tahapan Pengerjaan Proyek

1. **Import Library:** Mengimpor library yang dibutuhkan untuk pengolahan data, visualisasi, dan *machine learning*.
2. **Data Acquisition:** Memuat dataset Advertising dari URL ke dalam DataFrame.
3. **Data Understanding:** Melihat ukuran data, lima baris pertama, dan ringkasan statistik menggunakan `describe()`.
4. **Data Quality Check:** Memeriksa nilai kosong dan baris duplikat. Hasilnya, data bersih sehingga tidak diperlukan proses pembersihan tambahan.
5. **Exploratory Data Analysis (EDA):** Membuat *scatter plot* antara anggaran iklan di setiap media (TV, Radio, Newspaper) dan penjualan.
6. **Feature Selection:** Menggunakan `TV`, `radio`, dan `newspaper` sebagai fitur (X) dan `sales` sebagai target (y).
7. **Train-Test Split:** Membagi data menjadi 80% data latih (160 baris) dan 20% data uji (40 baris) dengan `random_state=42`.
8. **Model Training:** Melatih model Linear Regression menggunakan data latih.
9. **Model Evaluation:** Mengevaluasi model pada data uji menggunakan MAE, RMSE, dan R², serta visualisasi Prediksi vs Aktual dan distribusi residual.
10. **Prediction:** Menggunakan model untuk memprediksi penjualan pada tiga skenario anggaran iklan baru.

## Model Machine Learning

Algoritma yang digunakan dalam proyek ini adalah **Linear Regression**, yang mencari garis (bidang) linear terbaik untuk menghubungkan anggaran iklan dengan penjualan.

Persamaan model:

```
Sales = a₀ + a₁(TV) + a₂(Radio) + a₃(Newspaper)
```

## Performa Model

Evaluasi dilakukan menggunakan data uji sebanyak 40 baris yang tidak digunakan selama proses pelatihan.

| Metrik Evaluasi | Nilai | Keterangan |
|---|---:|---|
| MAE | 1,46 | Rata-rata selisih absolut prediksi dan nilai aktual (ribu unit) |
| RMSE | 1,78 | Seperti MAE, tetapi lebih sensitif terhadap kesalahan besar (ribu unit) |
| R² Score | 89,94% | Proporsi variasi penjualan yang dapat dijelaskan model |

Model mampu menjelaskan sekitar **89,9%** variasi penjualan hanya dari data anggaran iklan pada data uji.

Contoh perbandingan prediksi dan nilai aktual (lima data uji pertama):

| Aktual | Prediksi | Selisih |
|---:|---:|---:|
| 16,9 | 16,41 | -0,49 |
| 22,4 | 20,89 | -1,51 |
| 21,4 | 21,55 | 0,15 |
| 7,3 | 10,61 | 3,31 |
| 24,7 | 22,11 | -2,59 |

## Koefisien Model

Berdasarkan koefisien yang dipelajari model, diperoleh pengaruh setiap media iklan terhadap penjualan:

| Fitur | Koefisien |
|---|---:|
| radio | 0,1892 |
| TV | 0,0447 |
| newspaper | 0,0028 |
| Intercept | 2,9791 |

Koefisien menunjukkan perubahan penjualan (ribu unit) untuk setiap kenaikan anggaran sebesar $1.000 pada media tersebut, dengan asumsi media lain tetap. Radio memiliki koefisien terbesar per satuan anggaran, sedangkan koefisien Newspaper sangat kecil sehingga kontribusinya hampir tidak berarti dalam model ini.

Pada tahap visualisasi, *scatter plot* menunjukkan TV memiliki pola hubungan naik yang paling jelas terhadap penjualan, Radio juga berpengaruh cukup signifikan, sedangkan Newspaper paling lemah.

## Contoh Prediksi Skenario Baru

Model digunakan untuk memprediksi penjualan pada tiga skenario anggaran iklan:

| TV | Radio | Newspaper | Prediksi Sales (ribu unit) |
|---:|---:|---:|---:|
| 200 | 30 | 10 | 17,63 |
| 50 | 60 | 20 | 16,62 |
| 10 | 10 | 5 | 5,33 |

## Business Insight & Recommendation

Berdasarkan hasil EDA dan koefisien model Linear Regression, terdapat beberapa rekomendasi bisnis yang dapat dipertimbangkan oleh perusahaan:

1. **Memanfaatkan Model untuk Simulasi Anggaran**

   Perusahaan dapat menggunakan model ini untuk memperkirakan penjualan dari beberapa skenario alokasi anggaran sebelum biaya iklan dikeluarkan, sehingga keputusan alokasi dapat dipertimbangkan secara lebih terukur.

2. **Memprioritaskan Media dengan Pengaruh Lebih Besar**

   TV dan Radio menunjukkan hubungan yang lebih kuat dengan penjualan dibandingkan Newspaper. Perusahaan dapat mempertimbangkan untuk memprioritaskan kedua media tersebut, dengan tetap memperhatikan biaya dan jangkauan masing-masing media.

3. **Mengevaluasi Kembali Anggaran Koran**

   Kontribusi Newspaper pada model sangat kecil. Perusahaan dapat mengevaluasi efektivitas anggaran di media ini, misalnya dengan menguji model tanpa fitur Newspaper atau melakukan uji coba pengurangan anggaran secara terkontrol.

## Kesimpulan

Proyek ini menunjukkan penerapan *machine learning* dalam memprediksi penjualan berdasarkan anggaran iklan.

Model Linear Regression memperoleh R² sebesar 89,94%, MAE 1,46, dan RMSE 1,78 (ribu unit) pada data uji, yang menunjukkan bahwa model cukup baik dalam menjelaskan variasi penjualan. Hasil analisis juga menunjukkan bahwa Radio dan TV merupakan media dengan pengaruh lebih besar dibandingkan Newspaper.

Secara keseluruhan, proyek ini memberikan pengalaman dalam menerapkan tahapan analisis data dan *machine learning*, mulai dari pemuatan data, pemeriksaan kualitas data, visualisasi, pelatihan model, evaluasi, hingga penggunaan model untuk prediksi skenario baru.
