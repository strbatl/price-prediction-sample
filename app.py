import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load data
df = pd.read_excel("Dummy test for ML.xlsx")

# Preprocessing
df['IDR Price'] = df['IDR Price'].replace('[Rp,]', '', regex=True).str.replace('.', '', regex=False).astype(int)
df['SGD Price'] = df['SGD Price'].replace('[SGD,]', '', regex=True).str.replace(',', '', regex=False).astype(int)
df['Item Year'] = pd.to_datetime(df['Item Year'], errors='coerce').dt.year.fillna(0).astype(int)

label_cols = ['Brand Name', 'Type', 'Condition', 'Traders Name', 'Source']
label_encoders = {}
for col in label_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Train models
X = df[['Brand Name', 'Type', 'Condition', 'Item Year', 'Source']]
y_idr = df['IDR Price']
y_sgd = df['SGD Price']

model_idr = LinearRegression().fit(X, y_idr)
model_sgd = LinearRegression().fit(X, y_sgd)

# Streamlit UI
st.title("🕰️ Prediksi Harga Jam Tangan")

brand = st.selectbox("Brand", label_encoders['Brand Name'].classes_)
type_ = st.selectbox("Type", label_encoders['Type'].classes_)
condition = st.selectbox("Condition", label_encoders['Condition'].classes_)
item_year = st.number_input("Item Year", min_value=2000, max_value=2025, value=2024)
source = st.selectbox("Source", label_encoders['Source'].classes_)

if st.button("Prediksi Harga"):
    input_df = pd.DataFrame({
        'Brand Name': [label_encoders['Brand Name'].transform([brand])[0]],
        'Type': [label_encoders['Type'].transform([type_])[0]],
        'Condition': [label_encoders['Condition'].transform([condition])[0]],
        'Item Year': [item_year],
        'Source': [label_encoders['Source'].transform([source])[0]],
    })

    pred_idr = model_idr.predict(input_df)[0]
    pred_sgd = model_sgd.predict(input_df)[0]

    st.success(f"💵 Perkiraan Harga IDR: Rp {int(pred_idr):,}")
    st.success(f"💵 Perkiraan Harga SGD: SGD {int(pred_sgd):,}")
