import streamlit as st
import pandas as pd
import joblib

# ⚡ 必须第一个调用 Streamlit 命令
st.set_page_config(page_title="Auckland Properties Price Prediction", layout="centered")

# Load model and scaler
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load('knn_best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

model, scaler = load_model_and_scaler()

# Load suburb reference
@st.cache_data
def load_suburb_data():
    df_suburb_ref = pd.read_csv('suburb_ref.csv')
    suburb_data = df_suburb_ref.set_index('Suburb').to_dict(orient='index')
    return suburb_data

suburb_data = load_suburb_data()

# Property Type Mapping
property_type_map = {
    'House': 1,
    'Apartment': 2,
    'Townhouse': 3
}

# Streamlit UI
st.title("🏡 Auckland Properties Price Prediction App")

st.subheader("🔢 Input Property Information")

# User Inputs
bedrooms = st.number_input('Number of Bedrooms', min_value=1, max_value=10, value=3)
bathrooms = st.number_input('Number of Bathrooms', min_value=1, max_value=10, value=2)
property_type = st.selectbox('Select Property Type', options=list(property_type_map.keys()))
suburb = st.selectbox('Select Suburb', options=list(suburb_data.keys()))
st.write(f"(If you cannot find a suburb, it means that there are fewer than 5 properties currently for sale in that Suburb that are not predictable.)")
st.write(f"")

st.markdown("---")

# Show Suburb Info
suburb_info = suburb_data[suburb]
st.write(f"**Suburb Average Score:** {suburb_info['Score_avg']}")
st.write(f"(Suburb Average Score is based on the current price of the properties for sale in this suburb among all the properties for sale in Auckland, divided by the total number of properties for sale in this suburb. It has a certain suburb comprehensive reference value.)")
st.write(f"")
st.write(f"**Suburb Average Price:** ${int(suburb_info['avg_price']):,}")

st.markdown("---")

# Prediction
if st.button('🎯 Predict Property Price'):
    # Map property type to PropertyTypeID
    property_type_id = property_type_map[property_type]

    # Construct feature DataFrame
    input_features = pd.DataFrame([[
        bedrooms,
        bathrooms,
        suburb_info['Score_avg'],
        suburb_info['avg_price'],
        property_type_id
    ]], columns=[
        'Bedrooms', 'Bathrooms', 'Score_avg', 'avg_price', 'PropertyTypeID'
    ])

    # Scaling
    input_scaled = scaler.transform(input_features)

    # Prediction
    prediction = model.predict(input_scaled)

    st.success(f"🏠 Estimated Property Price: ${int(prediction[0]):,}")

st.markdown("---")
st.write("""
*Model: KNN Regressor (k=4), trained on Bedrooms, Bathrooms, Suburb Score, Suburb Average Price, and Property Type ID.*
""")
