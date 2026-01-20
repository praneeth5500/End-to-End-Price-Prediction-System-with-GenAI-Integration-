import streamlit as st
import pandas as pd
import pickle

# 1. Load the trained model
# Ensure 'car_model.pkl' is in the same folder
model = pickle.load(open("C:\\Users\\ppran\\OneDrive\\Desktop\\Data_analyst\\Used Car Price Valuation Tool\\Vscode\\car_model.pkl", 'rb'))

# 2. Define the lists of options (Copied from your training data)
brands = ['Acura', 'Alfa', 'Aston', 'Audi', 'BMW', 'Bentley', 'Buick', 'Cadillac', 'Chevrolet', 'Chrysler', 'Dodge', 'FIAT', 'Ferrari', 'Ford', 'GMC', 'Genesis', 'Honda', 'Hummer', 'Hyundai', 'INFINITI', 'Jaguar', 'Jeep', 'Karma', 'Kia', 'Lamborghini', 'Land', 'Lexus', 'Lincoln', 'Lotus', 'Lucid', 'MINI', 'Maserati', 'Maybach', 'Mazda', 'McLaren', 'Mercedes-Benz', 'Mercury', 'Mitsubishi', 'Nissan', 'Plymouth', 'Polestar', 'Pontiac', 'Porsche', 'RAM', 'Rivian', 'Rolls-Royce', 'Saab', 'Saturn', 'Scion', 'Subaru', 'Suzuki', 'Tesla', 'Toyota', 'Volkswagen', 'Volvo', 'smart']

# Top 50 models (as defined in your training step)
models = ['M3 Base', 'F-150 XLT', 'Corvette Base', '1500 Laramie', 'Wrangler Sport', 'Camaro 2SS', 'Model Y Long Range', '911 Carrera', 'Mustang GT Premium', 'M4 Base', 'Explorer XLT', 'F-250 Lariat', 'F-150 Lariat', 'E-Class E 350 4MATIC', 'M5 Base', 'E-Class E 350', '911 Carrera S', 'F-250 XLT', 'R1S Adventure Package', 'Land Cruiser Base', 'Macan S', 'ES 350 Base', 'Wrangler Unlimited Sport', 'Model 3 Long Range', 'GX 460 Base', 'Mustang GT', 'Corvette Stingray w/2LT', '1500 Big Horn', 'Model Y Performance', 'X7 xDrive40i', 'Cooper S Base', 'X5 xDrive35i', 'Highlander XLE', 'H2 Base', '911 Carrera 4S', 'Suburban Premier', 'Wrangler Unlimited Sahara', 'SL-Class SL 550', 'RX 350 Base', 'G-Class G 550 4MATIC', 'Suburban LT', 'C-Class C 300 4MATIC', 'GT-R Premium', 'X6 M Base', 'Camaro 1SS', 'Rover Range Rover Sport HSE', 'Tahoe LT', 'LX 570 Three-Row', 'CLA-Class CLA 250', '2500 Big Horn', 'Other']

fuel_types = ['Gasoline', 'Diesel', 'E85 Flex Fuel', 'Hybrid', 'Plug-In Hybrid', 'not supported']
transmissions = ['A/T', '8-Speed A/T', 'Transmission w/Dual Shift Mode', '6-Speed A/T', '6-Speed M/T', 'Automatic', '7-Speed A/T', '8-Speed Automatic', '10-Speed A/T', '5-Speed A/T', '9-Speed Automatic']

# 3. App Title & Description
st.title("🚗 Used Car Price Valuation Tool")
st.markdown("Enter the car details below to check if it's a **Fair Price**, **Good Deal**, or **Overpriced**.")

# 4. User Inputs (Layout with columns)
col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brands)
    model_choice = st.selectbox("Model", models)
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2019)
    mileage = st.number_input("Mileage (miles)", min_value=0, value=50000)

with col2:
    fuel = st.selectbox("Fuel Type", fuel_types)
    trans = st.selectbox("Transmission", transmissions)
    accident = st.radio("Has it been in an accident?", ["No", "Yes"])
    title = st.radio("Does it have a clean title?", ["Yes", "No"])

# Listing Price (For Comparison)
listing_price = st.number_input("Seller's Asking Price ($)", min_value=1000, value=25000)

# 5. Prediction Logic
if st.button("Analyze Deal"):
    # Preprocess inputs
    # Convert Yes/No to 1/0
    acc_val = 1 if accident == "Yes" else 0
    title_val = 1 if title == "Yes" else 0
    
    # Calculate Car Age
    car_age = 2025 - year
    
    # Create DataFrame for model
    input_data = pd.DataFrame({
        'brand': [brand],
        'model_simplified': [model_choice],
        'car_age': [car_age],
        'milage': [mileage],
        'fuel_type': [fuel],
        'transmission': [trans],
        'accident': [acc_val],
        'clean_title': [title_val]
    })
    
    # Predict
    predicted_price = model.predict(input_data)[0]
    
    # 6. Display Results
    st.subheader(f"Estimated Market Value: ${predicted_price:,.2f}")
    
    # Business Logic: Is it a good deal?
    diff = listing_price - predicted_price
    
    if diff < -1000:
        st.success(f" GREAT DEAL! You are saving ${abs(diff):,.2f} below market value.")
    elif diff > 1000:
        st.error(f"OVERPRICED. The seller is asking ${diff:,.2f} too much.")
    else:
        st.warning(f"FAIR PRICE. The listing is close to market value.")