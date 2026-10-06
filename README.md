# SMART-TILT Robo-Advisor

SMART-TILT adalah prototipe WealthTech Robo-Advisor berbasis algoritma kuantitatif yang memadukan valuasi Fama-French 5-Factor Model dengan optimasi portofolio Black-Litterman.

Aplikasi ini dirancang khusus untuk pasar saham Indonesia (IHSG), lengkap dengan fitur Realized Backtesting yang mempertimbangkan batasan diskrit pembelian saham (minimal 1 lot / 100 lembar) dan efek dana menganggur (Cash Drag).

## Fitur Utama
- **Fama-French Expected Return:** Mengestimasi imbal hasil intrinsik berdasarkan 5 faktor fundamental (Market, Size, Value, Profitability, Investment).
- **Black-Litterman Optimization:** Menyelaraskan keyakinan awal pasar (market prior) dengan pandangan subjektif klien (investor views).
- **Discrete Lot Allocation:** Secara otomatis mentranslasikan bobot matematis menjadi rekomendasi jumlah "Lot" yang siap dieksekusi di bursa saham nyata.
- **Risk Constraint Solver:** Algoritma SLSQP yang memastikan skenario kerugian maksimal portofolio tidak melewati batas toleransi profil risiko klien.

## Tech Stack
- **Python 3**
- **Frontend/UI:** Streamlit
- **Quantitative Engine:** SciPy (Solver), NumPy, Pandas
- **Econometrics (Backend):** Statsmodels (OLS Regression)
- **Data Visualization:** Matplotlib

## Struktur Repositori
- `app.py` — File utama yang berisi antarmuka pengguna Streamlit dan mesin algoritma.
- `Database.xlsx` — Basis data berisi riwayat harga saham, fundamental perusahaan, dan suku bunga bebas risiko (SBN).
- `Hasil_Regresi_Fama_French_5_Faktor.csv` — Hasil pre-kalkulasi regresi matriks valuasi fundamental.
- `requirements.txt` — Daftar dependensi modul Python yang dibutuhkan.

## Cara Menjalankan Aplikasi Lokal

1. **Clone repositori ini dan jalankan aplikasi:**
   ```bash
   git clone https://github.com/AOkta13/SMART-TILT.git
   cd SMART-TILT
   pip install -r requirements.txt
   python -m streamlit run app.py
   ```
2. **Atau dapat mengakses dari link:**
   ```bash
   https://tinyurl.com/Smart-Tilt
   ```