import streamlit as st
from src.predict import predict_price

st.set_page_config(page_title="House Price Prediction", page_icon="🏠", layout="centered")
st.title("🏠 House Price Prediction")
st.caption("Educational machine-learning demo. The included dataset is synthetic, not real market data.")

with st.form("prediction_form"):
    area = st.number_input("Built-up area (sq. ft.)", min_value=100, max_value=20000, value=1500, step=50)
    bedrooms = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)
    stories = st.number_input("Floors", min_value=1, max_value=5, value=2)
    parking = st.number_input("Parking spaces", min_value=0, max_value=5, value=1)
    location = st.selectbox("Location type", ["urban", "suburban", "rural"])
    condition = st.selectbox("Property condition", ["excellent", "good", "average", "needs_renovation"])
    submitted = st.form_submit_button("Predict price")

if submitted:
    try:
        value = predict_price({
            "area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
            "stories": stories, "parking": parking, "location": location,
            "condition": condition,
        })
        st.success(f"Estimated demonstration price: ₹{value:,.0f}")
        st.info("This is a model output from synthetic sample data, not a reliable real-world valuation.")
    except FileNotFoundError as error:
        st.error(str(error))
