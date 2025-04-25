import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Judul halaman
st.set_page_config(page_title="Prediksi Harga Jam Tangan", layout="centered")
st.title("🕰️ Prediksi Harga Jam Tangan Berdasarkan Data Historis")

# Load data
df = pd.read_excel("Dummy test for ML.xlsx")

# Bersihkan kolom harga (pastikan semuanya dalam bentuk string dulu)
df['IDR Price'] = df['IDR Price'].astype(str).replace('[Rp,]', '', regex=True).str.replace('.', '', regex=False).astype(int)
df['SGD Price'] = df['SGD Price'].astype(str).replace('[SGD,]', '', regex=True).str.replace(',', '', regex=False).astype(int)

# Ubah 'Item Year' jadi angka
df['Item Year'] = pd.to_datetime(df['Item Year'], errors='coerce').dt.year.fillna(0).astype(int)

# Encode fitur kategori
label_cols = ['Brand Name', 'Type', 'Condition', 'Traders Name', 'Source']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Siapkan fitur dan target
X = df[['Brand Name', 'Type', 'Condition', 'Item Year', 'Source']]
y_idr = df['IDR Price']
y_sgd = df['SGD Price']

# Model
model_idr = LinearRegression().fit(X, y_idr)
model_sgd = LinearRegression().fit(X, y_sgd)

# --- UI Streamlit ---
st.subheader("Masukkan Detail Produk")

brand = st.selectbox("Pilih Brand", label_encoders['Brand Name'].classes_)
type_ = st.selectbox("Tipe", label_encoders['Type'].classes_)
condition = st.selectbox("Kondisi", label_encoders['Condition'].classes_)
item_year = st.number_input("Tahun Produk", min_value=2000, max_value=2025, value=2024)
source = st.selectbox("Sumber", label_encoders['Source'].classes_)

if st.button("🔮 Prediksi Harga"):
    input_data = pd.DataFrame({
        'Brand Name': [label_encoders['Brand Name'].transform([brand])[0]],
        'Type': [label_encoders['Type'].transform([type_])[0]],
        'Condition': [label_encoders['Condition'].transform([condition])[0]],
        'Item Year': [item_year],
        'Source': [label_encoders['Source'].transform([source])[0]],
    })

    pred_idr = model_idr.predict(input_data)[0]
    pred_sgd = model_sgd.predict(input_data)[0]

    st.markdown("---")
    st.success(f"💰 Prediksi Harga IDR: Rp {int(pred_idr):,}")
    st.success(f"💰 Prediksi Harga SGD: SGD {int(pred_sgd):,}")

