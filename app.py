import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.dates as mdates
import plotly.graph_objects as go

# ==============================================================================
# 1. KONFIGURASI HALAMAN & UI/UX CSS
# ==============================================================================
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
# 2. PROSES ETL (Extract, Transform, Load) & CACHING
# ==============================================================================
@st.cache_data
def load_data():
    """Mengambil data dari Excel/CSV, membersihkan nama kolom, dan menghitung imbal hasil (return) bulanan."""
    try:
        df_meta = pd.read_excel('Database.xlsx', sheet_name='DB_ID_STOCKS').rename(columns=lambda x: str(x).strip())
        df_prices = pd.read_excel('Database.xlsx', sheet_name='P_ID_STOCKS').rename(columns=lambda x: str(x).strip())
        df_rf = pd.read_excel('Database.xlsx', sheet_name='IND_10Y_BOND').rename(columns=lambda x: str(x).strip())
        df_metrics = pd.read_csv('Hasil_Regresi_Fama_French_5_Faktor.csv', index_col='TICKER')
    except Exception:
        st.error("Gagal memuat data. Pastikan 'Database.xlsx' & 'Hasil_Regresi_Fama_French_5_Faktor.csv' ada di direktori yang sama.")
        st.stop()
        
    # Standarisasi waktu pada data harga saham & obligasi (SBN)
    df_prices['MONTH'] = pd.to_datetime(df_prices['MONTH'])
    df_prices = df_prices.sort_values('MONTH').reset_index(drop=True)
    df_rf.rename(columns={df_rf.columns[0]: 'MONTH'}, inplace=True)
    df_rf['MONTH'] = pd.to_datetime(df_rf['MONTH'])
    df_rf = df_rf.sort_values('MONTH').reset_index(drop=True)
    
    # Konversi Suku Bunga Bebas Risiko (RF) Tahunan ke Bulanan
    df_rf['IND_10Y_BOND'] = pd.to_numeric(df_rf['IND_10Y_BOND'], errors='coerce')
    df_rf['RF_ANNUAL'] = np.where(df_rf['IND_10Y_BOND'] > 1.0, df_rf['IND_10Y_BOND'] / 100.0, df_rf['IND_10Y_BOND'])
    df_rf['RF_MONTHLY'] = (1.0 + df_rf['RF_ANNUAL']) ** (1.0 / 12.0) - 1.0
    
    # Kalkulasi persentase perubahan harga bulanan (pct_change)
    valid_tickers = df_metrics.index.tolist()
    df_returns = df_prices[['MONTH', 'JKSE'] + valid_tickers].set_index('MONTH').pct_change().dropna()
    df_returns = df_returns.join(df_rf.set_index('MONTH')['RF_MONTHLY'], how='inner')
    
    return df_returns, df_metrics, df_meta.set_index('TICKERS'), df_prices

df_returns, df_metrics, df_meta, df_prices = load_data()

# ==============================================================================
# 3. ANTARMUKA APLIKASI (TABS)
# ==============================================================================
tab_optimasi, tab_screening = st.tabs(["📊 OPTIMASI PORTOFOLIO", "🔍 SCREENING SAHAM"])

# ------------------------------------------------------------------------------
# TAB 1: OPTIMASI PORTOFOLIO BLACK-LITTERMAN
# ------------------------------------------------------------------------------
with tab_optimasi:
    # --- A. Input Parameter Global ---
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown('<div class="metric-label">TOTAL INVESTASI (Rp)</div>', unsafe_allow_html=True)
        total_investasi = st.number_input("Total Investasi", min_value=0, value=100000000, step=10000000, format="%d", label_visibility="collapsed")
        st.markdown(f"<div style='color: #4CAF50; font-size: 15px; font-weight: bold; margin-top: -12px; margin-bottom: 10px;'>✅ Terbaca: Rp {total_investasi:,.0f}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-label">BATAS TOLERANSI RISIKO (%)</div>', unsafe_allow_html=True)
        batas_risiko = st.number_input("Batas Risiko", max_value=0, value=-10, step=1, label_visibility="collapsed")
    st.write("---")

    # --- B. State Management (Mencegah Reset Saat Mengetik) ---
    all_valid_tickers = df_metrics.index.tolist()
    if 'ui_kode_saham' not in st.session_state:
        st.session_state.ui_kode_saham = df_metrics.sort_values(by='MARKET_CAPITALIZATION', ascending=False).head(20).index.tolist()

    # --- C. Tabel Input Interaktif (Data Editor) ---
    df_ui = pd.DataFrame({
        'NO': range(1, 21),
        'KODE': st.session_state.ui_kode_saham,
        'NAMA PERUSAHAAN': pd.Series(st.session_state.ui_kode_saham).map(df_metrics['NAME']).fillna('Unknown'),
        'SEKTOR': pd.Series(st.session_state.ui_kode_saham).map(df_meta['SECTOR']).fillna('Unknown'),
        'INSIGHT VIEW NEXT MONTH (%)': [None] * 20, 
        'MINIMAL ALOKASI (%)': [None] * 20,
        'MAKSIMAL ALOKASI (%)': [None] * 20
    })

    edited_df = st.data_editor(
        df_ui, hide_index=True, use_container_width=True, num_rows="fixed", key="tabel_editor",
        column_config={
            "NO": st.column_config.NumberColumn(disabled=True),
            "KODE": st.column_config.SelectboxColumn(options=all_valid_tickers, required=True),
            "NAMA PERUSAHAAN": st.column_config.TextColumn(disabled=True),
            "SEKTOR": st.column_config.TextColumn(disabled=True),
            "INSIGHT VIEW NEXT MONTH (%)": st.column_config.NumberColumn(min_value=-100.0, max_value=100.0, step=0.1, format="%.2f"),
            "MINIMAL ALOKASI (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
            "MAKSIMAL ALOKASI (%)": st.column_config.NumberColumn(min_value=0.0, max_value=100.0, step=1.0, format="%.1f")
        }
    )

    # Deteksi perubahan susunan saham untuk re-render otomatis
    if st.session_state.ui_kode_saham != edited_df['KODE'].tolist():
        st.session_state.ui_kode_saham = edited_df['KODE'].tolist()
        st.rerun()

    # --- D. Eksekusi Engine Kuantitatif ---
    if st.button("JALANKAN OPTIMASI BLACK-LITTERMAN", type="primary", use_container_width=True):
        ui_tickers = edited_df['KODE'].tolist()
        
        # Validasi cegah duplikasi saham
        if len(ui_tickers) != len(set(ui_tickers)):
            st.error("⚠️ OPTIMASI GAGAL: Terdapat duplikasi Kode Saham pada tabel. Pastikan 20 saham yang dipilih berbeda!")
            st.stop()
            
        with st.spinner('Mesin matematika Black-Litterman sedang melakukan kalkulasi & konversi lot...'):
            num_assets = len(ui_tickers)
            
            # 1. Ekstraksi Pandangan Klien & Batasan Alokasi
            client_views = {row['KODE']: row['INSIGHT VIEW NEXT MONTH (%)'] / 100.0 for _, row in edited_df.dropna(subset=['INSIGHT VIEW NEXT MONTH (%)']).iterrows()}
            bounds = tuple((row['MINIMAL ALOKASI (%)'] / 100.0 if pd.notna(row['MINIMAL ALOKASI (%)']) else 0.0, 
                            row['MAKSIMAL ALOKASI (%)'] / 100.0 if pd.notna(row['MAKSIMAL ALOKASI (%)']) else 1.0) for _, row in edited_df.iterrows())
            client_max_loss = abs(batas_risiko / 100.0)
            
            # 2. Penyiapan Matriks Kovarians (Sigma) & Implied Equilibrium (Pi)
            df_ex_20 = df_returns[ui_tickers].sub(df_returns['RF_MONTHLY'], axis=0).dropna()
            jkse_ex = df_returns['JKSE'] - df_returns['RF_MONTHLY']
            lamb = max(jkse_ex.mean() / jkse_ex.var(), 2.5)
            W = df_metrics.loc[ui_tickers, 'MARKET_CAPITALIZATION'].fillna(1).values
            W = W / W.sum()
            Sigma = df_ex_20.cov().values
            Pi = lamb * (Sigma @ W)
            
            # 3. Injeksi Pandangan Fama-French & Klien (Matrix Q & Omega)
            tau = 1.0 / len(df_ex_20)
            Omega = np.diag(df_metrics.loc[ui_tickers, 'VARIANCE'].values)
            Q_series = df_metrics.loc[ui_tickers, 'EXPECTED_RETURN'].copy()
            Q_series.update(pd.Series(client_views))
            Q, P = Q_series.values, np.eye(num_assets)

            # 4. Master Formula Black-Litterman (Menghasilkan E_BL)
            inv_tau_Sigma = np.linalg.inv(tau * Sigma)
            inv_Omega = np.linalg.inv(Omega)
            E_BL = np.linalg.inv(inv_tau_Sigma + P.T @ inv_Omega @ P) @ (inv_tau_Sigma @ Pi + P.T @ inv_Omega @ Q)

            # 5. Optimasi Portofolio (SLSQP Solver) dengan Pembatasan Risiko
            rf_rate = df_returns['RF_MONTHLY'].iloc[-1]
            def neg_sharpe(w): return -(np.sum(w * E_BL) - rf_rate) / np.sqrt(w.T @ Sigma @ w)
            def max_loss_cons(w): return (np.sum(w * E_BL) - 2 * np.sqrt(w.T @ Sigma @ w)) + client_max_loss

            base_cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0}] 
            init_guess = np.full(num_assets, 1.0 / num_assets)
            opt_res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=base_cons + [{'type': 'ineq', 'fun': max_loss_cons}])
            
            # Fallback jika batas risiko terlalu sempit dan gagal ditemukan solusinya
            if not opt_res.success:
                st.warning(f"⚠️ Batas toleransi risiko **{batas_risiko}%** terlalu ketat. Algoritma mengabaikan batasan tersebut demi menemukan rasio Sharpe terbaik.")
                opt_res = minimize(neg_sharpe, init_guess, method='SLSQP', bounds=bounds, constraints=base_cons)
            optimal_weights = opt_res.x
            
            # 6. Translasi Bobot ke Lot Aktual Pasar Saham (Discrete Allocation)
            harga_saham = pd.to_numeric(df_meta.loc[ui_tickers, 'PRICE_TODAY'], errors='coerce').fillna(1).values
            rupiah_ideal = optimal_weights * total_investasi
            lot_aktual = np.where(optimal_weights < 0.0001, 0, np.floor(rupiah_ideal / (harga_saham * 100)))
            rupiah_aktual = lot_aktual * (harga_saham * 100)
            
            actual_weights = rupiah_aktual / total_investasi
            total_rupiah_aktual = np.sum(rupiah_aktual)
            sisa_cash = total_investasi - total_rupiah_aktual

            # 7. Kalkulasi Metrik Kinerja Portofolio Realistis
            port_ret = np.sum(actual_weights * E_BL)
            port_vol = np.sqrt(np.dot(actual_weights.T, np.dot(Sigma, actual_weights)))
            port_sharpe = (port_ret - rf_rate) / port_vol if port_vol != 0 else 0
            port_max_gain = port_ret + (2 * port_vol)
            port_max_loss_calc = max(port_ret - (2 * port_vol), -1.0) 

            # 8. Simulasi Backtesting 5 Tahun Terakhir
            hist_raw = df_prices.set_index('MONTH').sort_index().tail(61).pct_change().dropna(how='all')
            hist_port = hist_raw[ui_tickers].fillna(0).dot(actual_weights)
            cum_port, cum_mkt = (1 + hist_port).cumprod() - 1, (1 + hist_raw['JKSE'].fillna(0)).cumprod() - 1
            
            start_dt = hist_raw.index[0] - pd.DateOffset(months=1)
            cum_port.loc[start_dt], cum_mkt.loc[start_dt] = 0, 0
            cum_port, cum_mkt = cum_port.sort_index(), cum_mkt.sort_index()

            # --- E. Render Tabel Hasil Analisa Akhir ---
            st.markdown('<div class="hasil-header">HASIL & REKOMENDASI PEMBELIAN</div>', unsafe_allow_html=True)
            data_baris = []
            
            for i, ticker in enumerate(ui_tickers):
                w_act = actual_weights[i]
                m_loss = max(df_metrics.loc[ticker, 'MAX_LOSS'] if w_act == 0 else E_BL[i] - (2 * np.sqrt(Sigma[i, i])), -1.0)
                m_gain = df_metrics.loc[ticker, 'MAX_GAIN'] if w_act == 0 else E_BL[i] + (2 * np.sqrt(Sigma[i, i]))
                
                data_baris.append({
                    'NO': str(i + 1), 'KODE': ticker, 'NAMA PERUSAHAAN': df_metrics.loc[ticker, 'NAME'],
                    'HARGA/LEMBAR': f"Rp {harga_saham[i]:,.0f}" if harga_saham[i] > 1 else "-",
                    'BOBOT AKTUAL': f"{w_act * 100:.2f}%", 'REKOMENDASI BELI': f"{lot_aktual[i]:,.0f} Lot",
                    'DANA TERPAKAI': f"Rp {rupiah_aktual[i]:,.0f}" if rupiah_aktual[i] > 0 else "-",
                    'MAKSIMAL RUGI': f"{m_loss * 100:.2f}%", 'MAKSIMAL UNTUNG': f"{m_gain * 100:.2f}%"
                })
                
            # Menambahkan baris rekapitulasi ke dalam DataFrame
            df_display = pd.DataFrame(data_baris)
            baris_total = pd.DataFrame([{'NO': '', 'KODE': '', 'NAMA PERUSAHAAN': 'TOTAL DANA TERPAKAI (SAHAM)', 'HARGA/LEMBAR': '', 'BOBOT AKTUAL': f"{sum(actual_weights) * 100:.2f}%", 'REKOMENDASI BELI': '', 'DANA TERPAKAI': f"Rp {total_rupiah_aktual:,.0f}", 'MAKSIMAL RUGI': '', 'MAKSIMAL UNTUNG': ''}])
            baris_cash = pd.DataFrame([{'NO': '', 'KODE': 'CASH', 'NAMA PERUSAHAAN': 'SISA DANA (TIDAK CUKUP 1 LOT)', 'HARGA/LEMBAR': '', 'BOBOT AKTUAL': f"{(sisa_cash / total_investasi) * 100:.2f}%", 'REKOMENDASI BELI': '-', 'DANA TERPAKAI': f"Rp {sisa_cash:,.0f}", 'MAKSIMAL RUGI': '', 'MAKSIMAL UNTUNG': ''}])
            
            st.dataframe(pd.concat([df_display, baris_total, baris_cash], ignore_index=True), hide_index=True, use_container_width=True, height=820)

            # --- F. Render Grafik Interaktif Plotly ---
            st.write("<br><br>", unsafe_allow_html=True)
            
            fig = go.Figure()

            # Tambahkan Garis IHSG
            fig.add_trace(go.Scatter(
                x=cum_mkt.index, 
                y=cum_mkt,
                mode='lines', 
                name='IHSG',
                line=dict(color='#9E0A0F', width=3)
            ))

            # Tambahkan Garis Portfolio
            fig.add_trace(go.Scatter(
                x=cum_port.index, 
                y=cum_port,
                mode='lines', 
                name='PORTFOLIO AKTUAL (5 Thn)',
                line=dict(color='#2E8B57', width=3)
            ))

            # Konfigurasi Layout & Format Persentase
            fig.update_layout(
                title=dict(text="REALIZED PORTOFOLIO VS IHSG", font=dict(size=20)),
                xaxis=dict(showgrid=False, tickformat="%b-%Y"),
                yaxis=dict(showgrid=True, tickformat=".0%"),
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                margin=dict(l=0, r=0, t=50, b=0)
            )

            st.plotly_chart(fig, use_container_width=True, theme="streamlit")

            # --- G. Render Ringkasan Metrik ---
            st.markdown("---")
            st.markdown("**Metric (Based on Realized Allocation)**")

            metric_data = {
                "Indikator": [
                    "Expected Return Portofolio",
                    "Volatilitas Portofolio",
                    "Sharpe Ratio",
                    "Upside Potential",
                    "Downside Risk"
                ],
                "Nilai": [
                    f"{port_ret * 100:.2f}%",
                    f"{port_vol * 100:.2f}%",
                    f"{port_sharpe:.4f}",
                    f"{port_max_gain * 100:.2f}%",
                    f"{port_max_loss_calc * 100:.2f}%"
                ]
            }

            df_metrics = pd.DataFrame(metric_data)

            st.dataframe(df_metrics, hide_index=True, use_container_width=True)

# ------------------------------------------------------------------------------
# TAB 2: SCREENING SAHAM FUNDAMENTAL
# ------------------------------------------------------------------------------
with tab_screening:
    st.markdown('<div class="hasil-header">SCREENING FUNDAMENTAL & VALUASI SAHAM</div>', unsafe_allow_html=True)
    st.write("Gunakan fitur pencarian di bawah, atau arahkan kursor ke judul kolom tabel untuk memunculkan ikon Filter (seperti Excel).")
    
    # Menyiapkan kerangka data awal screening
    df_screen = df_metrics.copy()
        
    if df_screen.index.name:
        df_screen.index.name = str(df_screen.index.name).strip().upper()
    df_screen.columns = [str(col).strip().upper() for col in df_screen.columns]
        
    df_screen['SECTOR'] = df_screen.index.map(df_meta['SECTOR']).fillna('Unknown')
        
    df_screen = df_screen.reset_index()
        
    df_screen = df_screen.rename(columns={df_screen.columns[0]: 'TICKER'})
    
    kolom_wajib = ['RANKING', 'TICKER', 'NAME', 'SECTOR', 'MARKET_CAPITALIZATION', 'EXPECTED_RETURN', 'RISK', 'SHARPE_RATIO', 'MAX_GAIN', 'MAX_LOSS']
    df_screen = df_screen[[col for col in kolom_wajib if col in df_screen.columns]]

    # Konversi desimal menjadi persentase untuk estetika tabel
    for col in ['EXPECTED_RETURN', 'RISK', 'MAX_GAIN', 'MAX_LOSS']:
        if col in df_screen.columns:
            df_screen[col] = df_screen[col] * 100

    # UI Kotak Pencarian
    col_cari1, col_cari2, col_cari3 = st.columns(3)
    with col_cari1: cari_kode = st.text_input("🔍 Cari Kode (TICKER)", placeholder="Contoh: BBCA")
    with col_cari2: cari_nama = st.text_input("🏢 Cari Nama Perusahaan", placeholder="Contoh: Bank")
    with col_cari3: cari_sektor = st.selectbox("🏭 Filter Sektor", options=["Semua Sektor"] + sorted(df_screen['SECTOR'].unique().tolist()))

    # Logika Penyaringan (Filtering)
    if cari_kode: df_screen = df_screen[df_screen['TICKER'].str.contains(cari_kode.upper(), na=False)]
    if cari_nama: df_screen = df_screen[df_screen['NAME'].str.contains(cari_nama, case=False, na=False)]
    if cari_sektor != "Semua Sektor": df_screen = df_screen[df_screen['SECTOR'] == cari_sektor]

    st.caption(f"Menampilkan seluruh **{len(df_screen)}** saham yang cocok dengan kriteria pencarian Anda.")

    # Konversi ke skala Triliun Rupiah untuk visibilitas tabel rata kanan
    if 'MARKET_CAPITALIZATION' in df_screen.columns:
        df_screen['MARKET_CAPITALIZATION'] = df_screen['MARKET_CAPITALIZATION'] / 1_000_000_000_000
    
    # Eksekusi Render Tabel
    st.dataframe(
        df_screen, use_container_width=True, hide_index=True, height=700,
        column_config={
            "RANKING": st.column_config.NumberColumn("RANKING", format="%d"),
            "TICKER": st.column_config.TextColumn("KODE"),
            "NAMA": st.column_config.TextColumn("NAMA PERUSAHAAN"),
            "SECTOR": st.column_config.TextColumn("SEKTOR"),
            "MARKET_CAPITALIZATION": st.column_config.NumberColumn("KAPITALISASI (Triliun Rp)", format="%.2f T"), 
            "EXPECTED_RETURN": st.column_config.NumberColumn("EXPECTED RETURN", format="%.2f %%"),
            "RISK": st.column_config.NumberColumn("RISIKO (VOLATILITAS)", format="%.2f %%"),
            "SHARPE_RATIO": st.column_config.NumberColumn("SHARPE RATIO", format="%.2f"),
            "MAX_GAIN": st.column_config.NumberColumn("MAX GAIN", format="%.2f %%"),
            "MAX_LOSS": st.column_config.NumberColumn("MAX LOSS", format="%.2f %%"),
        }
    )