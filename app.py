import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates

# ==============================================================================
# 1. KONFIGURASI HALAMAN
# ==============================================================================
# Mengatur judul tab, lebar layar penuh (wide), dan menyuntikkan desain CSS kustom
st.set_page_config(page_title="SMART-TILT Robo-Advisor", layout="wide")
st.markdown("""
    <style>
    .smart-tilt-header { background-color: #9E0142; color: white; padding: 20px; text-align: center; font-size: 42px; font-weight: 900; letter-spacing: 2px; margin-bottom: 30px; }
    .hasil-header { background-color: #8A0F3D; color: white; padding: 10px; text-align: left; font-size: 24px; font-weight: 900; margin-top: 40px; margin-bottom: 20px; padding-left: 20px; }
    .metric-label { color: #FDAF61; font-size: 20px; font-weight: bold; }
    </style>
    <div class="smart-tilt-header">SMART-TILT</div>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. CACHE DATA (Memuat Data ke Memori 1x Saja)
# ==============================================================================
@st.cache_data
def load_data():
    try:
        # Membaca seluruh file database dan merapikan spasi tak terlihat pada nama kolom
        df_meta = pd.read_excel('Database.xlsx', sheet_name='DB_ID_STOCKS').rename(columns=lambda x: str(x).strip())
        df_prices = pd.read_excel('Database.xlsx', sheet_name='P_ID_STOCKS').rename(columns=lambda x: str(x).strip())
        df_rf = pd.read_excel('Database.xlsx', sheet_name='IND_10Y_BOND').rename(columns=lambda x: str(x).strip())
        df_metrics = pd.read_csv('Hasil_Regresi_Fama_French_5_Faktor.csv', index_col='TICKER')
    except Exception as e:
        st.error("Gagal memuat data. Pastikan 'Database.xlsx' dan 'Hasil_Regresi_Fama_French_5_Faktor.csv' berada di folder yang sama.")
        st.stop()
        
    # Standarisasi format waktu pada data harga saham
    df_prices['MONTH'] = pd.to_datetime(df_prices['MONTH'])
    df_prices = df_prices.sort_values('MONTH').reset_index(drop=True)
    
    # Standarisasi format waktu pada data Suku Bunga Bebas Risiko (Risk-Free / RF)
    df_rf.rename(columns={df_rf.columns[0]: 'MONTH'}, inplace=True)
    df_rf['MONTH'] = pd.to_datetime(df_rf['MONTH'])
    df_rf = df_rf.sort_values('MONTH').reset_index(drop=True)
    
    # De-compounding suku bunga RF tahunan menjadi imbal hasil bulanan murni
    df_rf['IND_10Y_BOND'] = pd.to_numeric(df_rf['IND_10Y_BOND'], errors='coerce')
    df_rf['RF_ANNUAL'] = np.where(df_rf['IND_10Y_BOND'] > 1.0, df_rf['IND_10Y_BOND'] / 100.0, df_rf['IND_10Y_BOND'])
    df_rf['RF_MONTHLY'] = (1.0 + df_rf['RF_ANNUAL']) ** (1.0 / 12.0) - 1.0
    
    # Menghitung imbal hasil (return) bulanan untuk setiap saham dan menggabungkannya dengan RF
    valid_tickers = df_metrics.index.tolist()
    df_returns = df_prices[['MONTH', 'JKSE'] + valid_tickers].set_index('MONTH').pct_change().dropna()
    df_returns = df_returns.join(df_rf.set_index('MONTH')['RF_MONTHLY'], how='inner')
    
    return df_returns, df_metrics, df_meta.set_index('TICKERS'), df_prices

# Menarik data dari fungsi cache agar siap digunakan oleh sistem
df_returns, df_metrics, df_meta, df_prices = load_data()

# ==============================================================================
# 3. INPUT GLOBAL KLIEN & UI TABEL REAKTIF
# ==============================================================================
# Menyusun tata letak 2 kolom untuk input Uang dan Batas Risiko
col1, col2 = st.columns([1, 1])
with col1:
    st.markdown('<div class="metric-label">TOTAL INVESTASI (Rp)</div>', unsafe_allow_html=True)
    total_investasi = st.number_input("", min_value=0, value=100000000, step=10000000, label_visibility="collapsed")
with col2:
    st.markdown('<div class="metric-label">BATAS TOLERANSI RISIKO (%)</div>', unsafe_allow_html=True)
    batas_risiko = st.number_input("", max_value=0, value=-10, step=1, label_visibility="collapsed")
st.write("---")

all_valid_tickers = df_metrics.index.tolist()

# Mengambil 20 saham kapitalisasi terbesar (Blue Chip) sebagai template default tabel
default_top_20 = df_metrics.sort_values(by='MARKET_CAPITALIZATION', ascending=False).head(20).index.tolist()

# Manajemen State (Memori) agar ketikan klien di tabel tidak hilang saat layar direfresh
if 'ui_data' not in st.session_state:
    st.session_state.ui_data = pd.DataFrame({
        'KODE': default_top_20,
        'INSIGHT VIEWS (%)': [None] * 20,
        'MINIMAL ALOKASI (%)': [None] * 20,
        'MAKSIMAL ALOKASI (%)': [None] * 20
    })

# Melakukan "Lookup" untuk menyinkronkan Kode Saham dengan Nama dan Sektornya
df_ui = st.session_state.ui_data.copy()
df_ui['NAMA PERUSAHAAN'] = df_ui['KODE'].map(df_metrics['NAME']).fillna('Unknown')
df_ui['SEKTOR'] = df_ui['KODE'].map(df_meta['SECTOR']).fillna('Unknown')
df_ui = df_ui[['KODE', 'NAMA PERUSAHAAN', 'SEKTOR', 'INSIGHT VIEWS (%)', 'MINIMAL ALOKASI (%)', 'MAKSIMAL ALOKASI (%)']]
df_ui.insert(0, 'NO', range(1, 21))

# Menampilkan antarmuka tabel interaktif (Data Editor)
edited_df = st.data_editor(
    df_ui, hide_index=True, use_container_width=True, num_rows="fixed",
    column_config={
        "NO": st.column_config.NumberColumn("NO", disabled=True),
        "KODE": st.column_config.SelectboxColumn("KODE", options=all_valid_tickers, required=True),
        "NAMA PERUSAHAAN": st.column_config.TextColumn("NAMA PERUSAHAAN", disabled=True),
        "SEKTOR": st.column_config.TextColumn("SEKTOR", disabled=True),
        "INSIGHT VIEWS (%)": st.column_config.NumberColumn("INSIGHT VIEWS (%)", min_value=-100.0, max_value=100.0, step=0.1, format="%.2f"),
        "MINIMAL ALOKASI (%)": st.column_config.NumberColumn("MINIMAL ALOKASI (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
        "MAKSIMAL ALOKASI (%)": st.column_config.NumberColumn("MAKSIMAL ALOKASI (%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f")
    }
)

# Detektor Pembaruan: Memaksa aplikasi refresh otomatis jika klien mengganti kode saham
if list(st.session_state.ui_data['KODE']) != list(edited_df['KODE']):
    st.session_state.ui_data = edited_df.copy()
    st.rerun()
else:
    st.session_state.ui_data = edited_df.copy()

# ==============================================================================
# 4. EKSEKUSI ENGINE & RENDER HASIL
# ==============================================================================
if st.button("JALANKAN OPTIMASI BLACK-LITTERMAN", type="primary", use_container_width=True):
    ui_tickers = edited_df['KODE'].tolist()
    
    # Validasi pencegahan input saham ganda
    if len(ui_tickers) != len(set(ui_tickers)):
        st.error("⚠️ OPTIMASI GAGAL: Terdapat duplikasi Kode Saham pada tabel input. Pastikan 20 saham yang dipilih seluruhnya berbeda!")
        st.stop()
        
    with st.spinner('Mesin matematika Black-Litterman sedang melakukan kalkulasi & konversi lot...'):
        
        # --- A. PENYIAPAN KENDALA (BOUNDS & VIEWS) ---
        num_assets = len(ui_tickers)
        # Mengekstrak input klien menjadi dictionary (Views) dan tuple (Bounds)
        client_views = {row['KODE']: row['INSIGHT VIEWS (%)'] / 100.0 for _, row in edited_df.dropna(subset=['INSIGHT VIEWS (%)']).iterrows()}
        bounds = tuple((row['MINIMAL ALOKASI (%)'] / 100.0 if pd.notna(row['MINIMAL ALOKASI (%)']) else 0.0, 
                        row['MAKSIMAL ALOKASI (%)'] / 100.0 if pd.notna(row['MAKSIMAL ALOKASI (%)']) else 1.0) for _, row in edited_df.iterrows())
        
        client_max_loss = abs(batas_risiko / 100.0)

        # --- B. MATRIKS PRIOR PASAR (Pi) ---
        df_ex_20 = df_returns[ui_tickers].sub(df_returns['RF_MONTHLY'], axis=0).dropna()
        jkse_ex = df_returns['JKSE'] - df_returns['RF_MONTHLY']
        
        # Parameter penolakan risiko (Risk Aversion) & Pembobotan Kapitalisasi
        lamb = max(jkse_ex.mean() / jkse_ex.var(), 2.5)
        W = df_metrics.loc[ui_tickers, 'MARKET_CAPITALIZATION'].fillna(1).values
        W = (W / W.sum())

        Sigma = df_ex_20.cov().values
        Pi = lamb * (Sigma @ W)
        tau = 1.0 / len(df_ex_20)
        Omega = np.diag(df_metrics.loc[ui_tickers, 'VARIANCE'].values)

        # --- C. PANDANGAN INTRINSIK (Q) ---
        # Mengadopsi valuasi Fama-French, ditimpa oleh input manual klien (jika ada)
        Q_series = df_metrics.loc[ui_tickers, 'EXPECTED_RETURN'].copy()
        Q_series.update(pd.Series(client_views))
        Q = Q_series.values
        P = np.eye(num_assets)

        # --- D. MASTER FORMULA BLACK-LITTERMAN ---
        # Menghitung imbal hasil ekspektasi kompromi (E_BL)
        inv_tau_Sigma = np.linalg.inv(tau * Sigma)
        inv_Omega = np.linalg.inv(Omega)
        E_BL = np.linalg.inv(inv_tau_Sigma + P.T @ inv_Omega @ P) @ (inv_tau_Sigma @ Pi + P.T @ inv_Omega @ Q)

        # --- E. ALGORITMA SOLVER (SciPy SLSQP) ---
        rf_rate = df_returns['RF_MONTHLY'].iloc[-1]
        def neg_sharpe(w): return -(np.sum(w * E_BL) - rf_rate) / np.sqrt(w.T @ Sigma @ w)
        def max_loss_cons(w): return (np.sum(w * E_BL) - 2 * np.sqrt(w.T @ Sigma @ w)) + client_max_loss

        base_cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}] # Wajib alokasi 100%
        init_guess = np.full(num_assets, 1.0 / num_assets)

        opt_res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=base_cons + [{'type': 'ineq', 'fun': max_loss_cons}])
        
        # Manajemen kegagalan konstrain (Notifikasi jika batas rugi terlalu sempit)
        if not opt_res.success:
            st.warning(f"⚠️ **PEMBERITAHUAN:** Batas toleransi risiko **{batas_risiko}%** terlalu ketat. Mesin mengabaikan batas kerugian tersebut dan tetap mencari portofolio dengan rasio Sharpe (keuntungan/risiko) paling optimal.")
            opt_res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=base_cons)

        optimal_weights = opt_res.x
        
        # --- F. TRANSLASI MATEMATIS KE LOT AKTUAL (Realized Portofolio) ---
        harga_saham = pd.to_numeric(df_meta.loc[ui_tickers, 'PRICE_TODAY'], errors='coerce').fillna(1).values
        rupiah_ideal = optimal_weights * total_investasi
        
        # Mengonversi target rupiah menjadi lot bulat kebawah menggunakan vektor NumPy
        lot_aktual = np.where(optimal_weights < 0.0001, 0, np.floor(rupiah_ideal / (harga_saham * 100)))
        rupiah_aktual = lot_aktual * (harga_saham * 100)
        
        actual_weights = rupiah_aktual / total_investasi
        total_rupiah_aktual = np.sum(rupiah_aktual)
        sisa_cash = total_investasi - total_rupiah_aktual

        # --- G. KALKULASI METRIK & GRAFIK BACKTESTING ---
        port_ret = np.sum(actual_weights * E_BL)
        port_vol = np.sqrt(np.dot(actual_weights.T, np.dot(Sigma, actual_weights)))
        port_sharpe = (port_ret - rf_rate) / port_vol if port_vol != 0 else 0
        port_max_gain = port_ret + (2 * port_vol)
        port_max_loss_calc = max(port_ret - (2 * port_vol), -1.0) # Proteksi batas rugi ekstrem di -100%

        # Mengekstrak riwayat murni 61 bulan terakhir (5 Tahun)
        hist_raw = df_prices.set_index('MONTH').sort_index().tail(61).pct_change().dropna(how='all')
        hist_saham_terpilih = hist_raw[ui_tickers].fillna(0) # Saham baru IPO dianggap return 0% di histori
        
        hist_port = hist_saham_terpilih.dot(actual_weights)
        cum_port = (1 + hist_port).cumprod() - 1
        cum_mkt = (1 + hist_raw['JKSE'].fillna(0)).cumprod() - 1
        
        # Menambahkan titik (0,0) di awal grafik agar presisi
        start_dt = hist_raw.index[0] - pd.DateOffset(months=1)
        cum_port.loc[start_dt], cum_mkt.loc[start_dt] = 0, 0
        cum_port, cum_mkt = cum_port.sort_index(), cum_mkt.sort_index()

        # --- H. MERAKIT TABEL HASIL REKOMENDASI ---
        st.markdown('<div class="hasil-header">HASIL & REKOMENDASI PEMBELIAN</div>', unsafe_allow_html=True)
        data_baris = []
        
        for i, ticker in enumerate(ui_tickers):
            w_act = actual_weights[i]
            
            # Memastikan tidak ada prediksi kerugian individu melampaui -100%
            kalkulasi_rugi = df_metrics.loc[ticker, 'MAX_LOSS'] if w_act == 0 else E_BL[i] - (2 * np.sqrt(Sigma[i, i]))
            m_loss = max(kalkulasi_rugi, -1.0)
            m_gain = df_metrics.loc[ticker, 'MAX_GAIN'] if w_act == 0 else E_BL[i] + (2 * np.sqrt(Sigma[i, i]))
            
            data_baris.append({
                'NO': str(i + 1), 
                'KODE': ticker, 
                'NAMA PERUSAHAAN': df_metrics.loc[ticker, 'NAME'],
                'HARGA/LEMBAR': f"Rp {harga_saham[i]:,.0f}" if harga_saham[i] > 1 else "-",
                'BOBOT AKTUAL': f"{w_act * 100:.2f}%",
                'REKOMENDASI BELI': f"{lot_aktual[i]:,.0f} Lot",
                'DANA TERPAKAI': f"Rp {rupiah_aktual[i]:,.0f}" if rupiah_aktual[i] > 0 else "-",
                'MAKSIMAL RUGI': f"{m_loss * 100:.2f}%", 
                'MAKSIMAL UNTUNG': f"{m_gain * 100:.2f}%"
            })
            
        df_display = pd.DataFrame(data_baris)
        
        # Menambahkan baris penutup (Total dan Uang Kas)
        total_persen_aktual = sum(actual_weights) * 100
        baris_total = pd.DataFrame([{ 
            'NO': '', 'KODE': '', 'NAMA PERUSAHAAN': 'TOTAL DANA TERPAKAI (SAHAM)', 
            'HARGA/LEMBAR': '', 'BOBOT AKTUAL': f"{total_persen_aktual:.2f}%", 'REKOMENDASI BELI': '',
            'DANA TERPAKAI': f"Rp {total_rupiah_aktual:,.0f}", 'MAKSIMAL RUGI': '', 'MAKSIMAL UNTUNG': '' 
        }])
        
        baris_cash = pd.DataFrame([{ 
            'NO': '', 'KODE': 'CASH', 'NAMA PERUSAHAAN': 'SISA DANA (TIDAK CUKUP 1 LOT)', 
            'HARGA/LEMBAR': '', 'BOBOT AKTUAL': f"{(sisa_cash / total_investasi) * 100:.2f}%", 'REKOMENDASI BELI': '-',
            'DANA TERPAKAI': f"Rp {sisa_cash:,.0f}", 'MAKSIMAL RUGI': '', 'MAKSIMAL UNTUNG': '' 
        }])
        
        df_display = pd.concat([df_display, baris_total, baris_cash], ignore_index=True)
        st.dataframe(df_display, hide_index=True, use_container_width=True, height=820)

        # --- I. RENDER GRAFIK (MATPLOTLIB) & METRIK SUMMARY ---
        st.write("<br><br>", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(14, 6))
        bg_color = '#0E1117'
        fig.patch.set_facecolor(bg_color) 
        ax.set_facecolor(bg_color)        

        ax.plot(cum_mkt.index, cum_mkt, color='#9E0A0F', linewidth=3, label='IHSG')       
        ax.plot(cum_port.index, cum_port, color='#2E8B57', linewidth=3, label='PORTFOLIO AKTUAL (5 Thn)') 

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)) 
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b-%Y'))
        ax.tick_params(colors='#D3D3D3', labelsize=9)
        plt.xticks(rotation=45, ha='right')

        ax.grid(axis='y', linestyle='-', color='#333333', linewidth=1) 
        ax.grid(axis='x', visible=False)
        for spine in ['top', 'right', 'left']: ax.spines[spine].set_visible(False)
        ax.spines['bottom'].set_color('#D3D3D3')

        plt.title('REALIZED PORTOFOLIO VS IHSG', fontsize=16, color='white', pad=20)
        legend = plt.legend(loc='lower center', bbox_to_anchor=(0.5, -0.3), ncol=2, frameon=False, fontsize=11)
        for text in legend.get_texts(): text.set_color("white")
        
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown("---")
        col_met_1, col_met_2 = st.columns([1, 2])
        with col_met_1:
            st.markdown(f"""
            <div style="color: white; font-size: 16px; line-height: 1.8;">
                <b>Metric (Based on Realized Allocation)</b><br>
                Expected Return Portofolio <span style="float:right;">{port_ret * 100:.2f}%</span><br>
                Volatilitas Portofolio <span style="float:right;">{port_vol * 100:.2f}%</span><br>
                Sharpe Ratio <span style="float:right;">{port_sharpe:.4f}</span><br>
                Upside Potential <span style="float:right;">{port_max_gain * 100:.2f}%</span><br>
                Downside Risk <span style="float:right;">{port_max_loss_calc * 100:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)