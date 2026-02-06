# 🚗 Used Car Price Valuation Tool

A machine learning application that estimates the market value of used cars based on factors like brand, model, year, mileage, and accident history.

## 📊 Overview

This project analyzes a dataset of used cars to understand price depreciation and market trends. It uses a Random Forest Regressor to predict prices and presents the model via an interactive web application built with Streamlit.

**Key Features:**
- **Data Cleaning:** Handling missing values, outlier removal, and string formatting.
- **Feature Engineering:** Calculating vehicle age and simplifying model names.
- **Machine Learning:** Trained using `RandomForestRegressor` with a pipeline for encoding categorical data.
- **Web App:** A user-friendly interface to input car details and get an instant price prediction.

## 🛠️ Tech Stack

- **Python** (Logic & Data Analysis)
- **Pandas & NumPy** (Data Manipulation)
- **Scikit-Learn** (Machine Learning)
- **Streamlit** (Web Framework)

## 📂 Project Structure

- `app.py`: The main Streamlit application script.
- `notebooks/`: Jupyter notebooks used for EDA (Exploratory Data Analysis) and model training.
- `model/`: Contains the serialized `.pkl` machine learning model.
- `data/`: The raw `used_cars.csv` dataset.

## 🚀 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/car-price-predictor.git](https://github.com/your-username/car-price-predictor.git)
   cd car-price-predictor