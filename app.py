# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.api import SimpleExpSmoothing, Holt, ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import pmdarima as pm
import os
import re
import warnings

# Mengabaikan peringatan teknis agar output lebih bersih
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8-whitegrid')

# =============================================================================
# 1. FUNGSI UNTUK MEMUAT DAN MEMBERSIHKAN DATA
# =============================================================================
@st.cache_data
def load_data():
    """
    Fungsi ini mencari file CSV di folder yang sama, membacanya,
    membersihkan, dan mengubahnya menjadi format deret waktu yang siap dianalisis.
    """
    folder_path = "."  # Asumsi file CSV berada di folder yang sama dengan app.py
    file_names = sorted([
        f for f in os.listdir(folder_path)
        if f.startswith("Tingkat Pengangguran Terbuka") and f.endswith(".csv")
    ])

    if not file_names:
        st.error("Tidak ada file CSV 'Tingkat Pengangguran Terbuka' yang ditemukan di folder ini.")
        return {}

    records = []
    for file_name in file_names:
        try:
            match = re.search(r'(\d{4})', file_name)
            if not match:
                continue
            year = int(match.group(1))

            df_raw = pd.read_csv(os.path.join(folder_path, file_name), skiprows=2)
            df_raw.rename(columns={
                df_raw.columns[0]: 'Pendidikan',
                df_raw.columns[1]: 'Februari',
                df_raw.columns[2]: 'Agustus'
            }, inplace=True)

            df_data = df_raw[['Pendidikan', 'Februari', 'Agustus']].iloc[1:8].copy()
            df_data['Februari'] = pd.to_numeric(df_data['Februari'], errors='coerce')
            df_data['Agustus'] = pd.to_numeric(df_data['Agustus'], errors='coerce')

            for _, row in df_data.iterrows():
                records.append({
                    'tahun': year, 'semester': 1,
                    'pendidikan': row['Pendidikan'], 'tpt': row['Februari']
                })
                records.append({
                    'tahun': year, 'semester': 2,
                    'pendidikan': row['Pendidikan'], 'tpt': row['Agustus']
                })
        except Exception as e:
            st.warning(f"Gagal memproses file {file_name}: {e}")

    if not records:
        st.error("Tidak dapat memuat catatan apa pun dari file CSV. Pastikan nama file dan format internalnya benar.")
        return {}

    df_final = pd.DataFrame(records)

    # Menghapus data anomali Februari 2019
    df_final = df_final[~((df_final['tahun'] == 2019) & (df_final['semester'] == 1))]

    # Membuat index waktu yang benar
    df_final['periode'] = pd.PeriodIndex(
        year=df_final['tahun'],
        quarter=(df_final['semester'] - 1) * 2 + 2,
        freq='Q-DEC'
    )
    df_final.set_index('periode', inplace=True)

    # Membersihkan data dari nilai kosong dan data yang tidak lengkap
    df_final.dropna(subset=['tpt'], inplace=True)
    df_final = df_final[~((df_final['tahun'] >= 2025) & (df_final['semester'] == 2) & pd.isna(df_final['tpt']))]

    # Normalisasi kategori pendidikan untuk konsistensi
    replace_map = {
        'SD ke Bawah': 'SD', 'Tamat SD': 'SD', 'Tidak/Belum Pernah Sekolah': 'SD',
        'SMP': 'SMP', 'SMA': 'SMA', 'SMA Kejuruan': 'SMK', 'SMK': 'SMK',
        'Diploma I/II/III': 'Diploma', 'Universitas': 'Universitas'
    }
    df_final['pendidikan'] = df_final['pendidikan'].str.strip().replace(replace_map)

    # Memisahkan data per kategori
    kategori_list = ['SD', 'SMP', 'SMA', 'SMK', 'Diploma', 'Universitas']
    all_series = {}
    for kat in kategori_list:
        series = df_final[df_final['pendidikan'] == kat]['tpt'].dropna()
        if not series.empty:
            all_series[kat] = series.sort_index()

    return all_series


# =============================================================================
# 2. FUNGSI UTAMA UNTUK ANALISIS
# =============================================================================
def run_analysis(time_series):
    # --- Melatih semua model terlebih dahulu ---
    with st.spinner("Melatih dan mengevaluasi semua model..."):
        ses_fit = SimpleExpSmoothing(time_series, initialization_method="estimated").fit()
        des_fit = Holt(time_series, initialization_method="estimated").fit()
        tes_fit = ExponentialSmoothing(
            time_series,
            trend='add',
            seasonal='add',
            seasonal_periods=2,
            initialization_method="estimated"
        ).fit()
        auto_model = pm.auto_arima(
            time_series, seasonal=True, m=2,
            suppress_warnings=True, stepwise=True, trace=False
        )
        sarima_fit = SARIMAX(
            time_series,
            order=auto_model.order,
            seasonal_order=auto_model.seasonal_order,
            enforce_stationarity=False
        ).fit(disp=False)

        preds = {
            'SES': ses_fit.fittedvalues,
            'DES': des_fit.fittedvalues,
            'TES': tes_fit.fittedvalues,
            'SARIMA': sarima_fit.get_prediction(start=0).predicted_mean
        }

        def eval_metric(y_true, y_pred):
            y_true = y_true[y_pred.notna()]
            y_pred = y_pred.dropna()
            return [
                mean_absolute_percentage_error(y_true, y_pred) * 100,
                mean_squared_error(y_true, y_pred),
                mean_absolute_error(y_true, y_pred)
            ]

        metrics = {name: eval_metric(time_series, pred) for name, pred in preds.items()}
        df_metrics = pd.DataFrame(metrics, index=['MAPE (%)', 'MSE', 'MAD']).T.reset_index()
        df_metrics.columns = ['Metode', 'MAPE (%)', 'MSE', 'MAD']

        best_model_row = df_metrics.loc[df_metrics['MAPE (%)'].idxmin()]
        best_model_name = best_model_row['Metode']
        best_model_mape = best_model_row['MAPE (%)']

        n_forecast = 1
        best_model_fit = {
            'SES': ses_fit,
            'DES': des_fit,
            'TES': tes_fit,
            'SARIMA': sarima_fit
        }[best_model_name]
        if best_model_name == 'SARIMA':
            next_forecast = best_model_fit.get_forecast(steps=n_forecast).predicted_mean.iloc[0]
        else:
            next_forecast = best_model_fit.forecast(steps=n_forecast).iloc[0]

    # --- Dashboard Ringkasan Hasil ---
    st.subheader("📊 Dashboard Ringkasan")
    col1, col2, col3 = st.columns(3)
    col1.metric("Model Terbaik", best_model_name)
    col2.metric("MAPE (%) Model Terbaik", f"{best_model_mape:.2f}%")
    col3.metric("Prediksi 2025-S2", f"{next_forecast:.2f}%")

    st.markdown("---")

    # --- Detail teknis: Uji Stasioneritas dan ACF/PACF ---
    with st.expander("Lihat Detail Teknis (Uji Stasioneritas dan ACF/PACF)"):
        st.subheader("Uji Stasioneritas dan Analisis Korelasi")
        adf_result = adfuller(time_series)
        st.write(f"**a. Uji Stasioneritas (ADF Test)**: p-value = {adf_result[1]:.4f}")
        if adf_result[1] > 0.05:
            st.warning("Data tidak stasioner. Differencing dilakukan untuk analisis ACF/PACF.")
            ts_diff = time_series.diff(1).diff(2).dropna()
        else:
            st.success("Data sudah stasioner.")
            ts_diff = time_series

        st.write("**b. Plot ACF dan PACF (untuk identifikasi model)**")
        max_lags = len(ts_diff) // 2 - 1
        if max_lags < 1:
            max_lags = 1
        fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
        plot_acf(ts_diff, ax=ax1, lags=max_lags)
        ax1.set_title("Autocorrelation Function (ACF)")
        plot_pacf(ts_diff, ax=ax2, lags=max_lags)
        ax2.set_title("Partial Autocorrelation Function (PACF)")
        st.pyplot(fig2)

    # --- Evaluasi Kinerja ---
    st.subheader("📈 Hasil Pemodelan & Evaluasi Kinerja")
    st.write("**Tabel Perbandingan Kinerja Model**")
    st.dataframe(df_metrics.round(3))

    st.write("**Visualisasi Perbandingan Kinerja Model**")
    df_metrics_plot = df_metrics.set_index('Metode')
    fig4, ax4 = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    metrics_to_plot = ['MAPE (%)', 'MSE', 'MAD']
    method_colors = {'SES': '#6495ED', 'DES': '#3CB371', 'TES': '#F08080', 'SARIMA': '#9370DB'}
    for i, metric in enumerate(metrics_to_plot):
        bars = df_metrics_plot[metric].plot(
            kind='bar', ax=ax4[i],
            color=[method_colors.get(x, 'gray') for x in df_metrics_plot.index]
        )
        ax4[i].set_title(f'Perbandingan {metric}')
        ax4[i].set_xlabel('')
        ax4[i].tick_params(axis='x', rotation=45)
        for bar in bars.patches:
            bars.annotate(
                f'{bar.get_height():.2f}',
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha='center', va='bottom',
                xytext=(0, 5), textcoords='offset points'
            )
        ax4[i].set_ylim(top=ax4[i].get_ylim()[1] * 1.15)
    plt.tight_layout()
    st.pyplot(fig4)

    # --- Parameter Optimal ---
    st.subheader("🔍 Analisis Parameter Optimal")
    st.write("**Parameter Optimal Exponential Smoothing**")
    params_data = {
        'Metode': ['SES', 'DES', 'TES'],
        'Alpha (Level)': [
            ses_fit.params.get('smoothing_level'),
            des_fit.params.get('smoothing_level'),
            tes_fit.params.get('smoothing_level')
        ],
        'Beta (Trend)': [
            None,
            des_fit.params.get('smoothing_trend'),
            tes_fit.params.get('smoothing_trend')
        ],
        'Gamma (Seasonal)': [
            None,
            None,
            tes_fit.params.get('smoothing_seasonal')
        ]
    }
    params_df = pd.DataFrame(params_data)

    def smart_format(val):
        if pd.isna(val):
            return '-'
        if abs(val) < 0.001 and val != 0:
            return f"{val:.2e}"
        return f"{val:.3f}"

    st.dataframe(params_df.style.format({
        'Alpha (Level)': smart_format,
        'Beta (Trend)': smart_format,
        'Gamma (Seasonal)': smart_format
    }))
    st.info(f"**Model SARIMA Terbaik:** {auto_model.order} x {auto_model.seasonal_order}")

    # --- TABEL NILAI AKTUAL vs PREDIKSI (In-Sample) ---
    st.subheader("📑 Tabel Nilai Aktual vs Prediksi (In-Sample)")
    df_compare = pd.DataFrame(index=time_series.index)
    df_compare['Aktual'] = time_series
    for name, pred in preds.items():
        df_compare[name] = pred.reindex(time_series.index)

    show_errors = st.checkbox("", value=False)
    if show_errors:
        for name in ['SES', 'DES', 'TES', 'SARIMA']:
            ae_col = f'AE_{name}'
            ape_col = f'APE_{name} (%)'
            df_compare[ae_col] = (df_compare[name] - df_compare['Aktual']).abs()
            denom = df_compare['Aktual'].replace(0, np.nan)
            df_compare[ape_col] = (df_compare[ae_col] / denom) * 100

    df_show = df_compare.copy()
    df_show.index = df_show.index.astype(str)  # tampilkan Period sebagai string
    st.dataframe(df_show.round(3))

    csv_bytes = df_show.to_csv(index_label='Periode').encode('utf-8')
    st.download_button("⬇️ Unduh Tabel Prediksi (CSV)",
                       data=csv_bytes,
                       file_name="tabel_prediksi_in_sample.csv",
                       mime="text/csv")

    # --- Visualisasi Prediksi In-Sample ---
    st.subheader("📉 Visualisasi Prediksi In-Sample")
    fig3, ax3 = plt.subplots(figsize=(12, 6))

    color_map = {
        'SES': '#1f77b4',
        'DES': '#2ca02c',
        'TES': '#d62728',
        'SARIMA': '#9467bd'
    }

    # Plot data aktual
    ax3.plot(
        time_series.index.to_timestamp(), time_series,
        marker='o', color='black', linewidth=3,
        label='Data Aktual', zorder=10
    )

    # Plot tiap model
    for name, pred in preds.items():
        color = color_map.get(name, 'gray')
        linewidth = 3 if name == best_model_name else 1.5
        linestyle = '--' if name != best_model_name else '-'
        label = f'{name} (Terbaik)' if name == best_model_name else name

        ax3.plot(
            pred.index.to_timestamp(), pred,
            label=label, linestyle=linestyle,
            linewidth=linewidth, color=color
        )

    ax3.set_title("Perbandingan Prediksi dengan Data Aktual", fontsize=16)
    ax3.set_xlabel("Tahun", fontsize=12)
    ax3.set_ylabel("TPT (%)", fontsize=12)
    ax3.tick_params(labelsize=11)
    ax3.grid(True, linestyle='--', alpha=0.6)
    ax3.legend(title="Keterangan", fontsize=10, title_fontsize=11, loc='upper right')
    st.pyplot(fig3)


# =============================================================================
# 3. TAMPILAN APLIKASI GUI STREAMLIT
# =============================================================================
st.set_page_config(layout="wide")
st.title("📊 Aplikasi Prediksi Tingkat Pengangguran Terbuka (TPT) Jawa Barat")
st.write("Aplikasi ini membandingkan 4 metode peramalan (SES, DES, TES, SARIMA) pada data TPT semesteran per tingkat pendidikan.")

try:
    data_all = load_data()
    if not data_all:
        st.info("Menunggu file data CSV diunggah ke folder...")
        st.stop()

    st.sidebar.header("⚙️ Kontrol Analisis")
    kategori = st.sidebar.selectbox("Pilih Tingkat Pendidikan:", options=list(data_all.keys()))

    if st.sidebar.button("Jalankan Analisis Model"):
        st.header(f"Hasil Analisis TPT untuk Tingkat Pendidikan: {kategori}")
        st.subheader("📖 Data Historis yang Digunakan")
        display_df = data_all[kategori].to_frame().reset_index()
        display_df.columns = ['Periode', 'TPT (%)']
        display_df['Periode'] = display_df['Periode'].astype(str)
        st.dataframe(display_df.style.format({'TPT (%)': '{:.2f}'}))

        run_analysis(data_all[kategori])
except Exception as e:
    st.error(f"Terjadi kesalahan saat menjalankan aplikasi: {e}")
