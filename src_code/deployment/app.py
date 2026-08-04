import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="naveenaggarwal1989/superkart-sales-model", filename="best_superkart_sales_model_v1.joblib")
model = joblib.load(model_path)

# Streamlit UI for SuperKart Sales Prediction
st.title("Hey User, Welcome!!")
st.title("SuperKart Products Sale Forecast API")
st.write("""
This application predicts estimated product sales for a store.
""")

# User input widgets for each feature
with st.form("Sales Estimation"):
    # Product Inputs with min/max constraints based on EDA
    # Category to Prefix mapping
    CATEGORY_PREFIX_MAP = {
        'Hard Drinks': 'DR',
        'Soft Drinks': 'DR',
        'Frozen Foods': 'FD',
        'Dairy': 'FD',
        'Canned': 'FD',
        'Baking Goods': 'FD',
        'Snack Foods': 'FD',
        'Meat': 'FD',
        'Fruits and Vegetables': 'FD',
        'Breads': 'FD',
        'Breakfast': 'FD',
        'Starchy Foods': 'FD',
        'Seafood': 'FD',
        'Health and Hygiene': 'NC',
        'Household': 'NC',
        'Others': 'NC'
    }

    product_categories = list(CATEGORY_PREFIX_MAP.keys())

    # Store Specification
    st.subheader("Enter Details Below for Product Sale Forecast")
    st.markdown("#### 1. Store Details")
    col1, col2 = st.columns(2)
    with col1:
        store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type 1", "Supermarket Type2", "Food Mart"])
        store_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    with col2:
        city_type = st.selectbox("Store City Location", ["Tier 1", "Tier 2", "Tier 3"])
        store_year = st.number_input("Store Establishment Year", min_value=1987, max_value=2009, value=2009)

    # Product Specification
    st.markdown("#### 2. Product Details")
    col1, col2 = st.columns(2)
    with col1:
        product_type = st.selectbox("Product Type", product_categories)
        product_mrp = st.number_input("Product MRP", min_value=5.0, max_value=500.0, value=117.08)
        product_weight = st.number_input("Product Weight", min_value=3.0, max_value=30.0, value=12.66)
    with col2:
        product_id = st.text_input("Product ID", value="FD306")
        sugar_content = st.selectbox("Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
        allocated_area = st.number_input("Allocated Area", min_value=0.0001, max_value=0.3, value=0.027, format="%.3f")
    submitted = st.form_submit_button("Predict Purchase")

    if submitted:
        # Assemble input into DataFrame, ensuring correct column order and types for the model
        input_data = pd.DataFrame([
            {
                'Product_Weight': product_weight,
                'Product_Sugar_Content': sugar_content,
                'Product_Allocated_Area': allocated_area,
                'Product_Type': product_type,
                'Product_MRP': product_mrp,
                'Store_Establishment_Year': store_year,
                'Store_Size': store_size,
                'Store_Location_City_Type': city_type,
                'Store_Type': store_type
            }
        ])

        predicted_sales = model.predict(input_data)[0]

        st.subheader("Prediction Result:")
        st.success(f"The model predicts estimated sales of: **${predicted_sales:.2f}**")
