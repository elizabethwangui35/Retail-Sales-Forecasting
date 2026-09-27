import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model
model = joblib.load("final_model.pkl")

# Load model feature information
model_columns = joblib.load("model_columns.pkl")

# Load historical sales data
sales_history = pd.read_csv("sales_history_app.csv")
sales_history["date"] = pd.to_datetime(sales_history["date"])

# Load store information
stores = pd.read_csv("stores_app.csv")

# Load holiday information
holidays = pd.read_csv("holidays_app.csv")
holidays["date"] = pd.to_datetime(holidays["date"])

# Load oil price information
oil = pd.read_csv("oil_app.csv")
oil["date"] = pd.to_datetime(oil["date"])

# Application title
st.title("Retail Sales Forecasting App")

st.write(
    "Use this application to estimate future retail sales "
    "for a selected store and product family."
)

# User inputs
st.subheader("Enter Forecast Information")

forecast_date = st.date_input(
    "Forecast Date",
    value=pd.to_datetime("2017-08-16"),
    min_value=pd.to_datetime("2017-08-16"),
    max_value=pd.to_datetime("2017-08-31")
)

store_number = st.selectbox(
    "Store Number",
    sorted(stores["store_nbr"].unique())
)

product_family = st.selectbox(
    "Product Family",
    sorted(sales_history["family"].unique())
)

onpromotion = st.number_input(
    "Number of Products on Promotion",
    min_value=0,
    value=0,
    step=1
)

# Convert date
forecast_date = pd.to_datetime(forecast_date)

# Retrieve store information
selected_store = stores[
    stores["store_nbr"] == store_number
].iloc[0]

store_type = selected_store["type"]
cluster = selected_store["cluster"]

# Generate date features
year = forecast_date.year
month = forecast_date.month
day = forecast_date.day
day_of_week_num = forecast_date.dayofweek
is_weekend = int(day_of_week_num in [5, 6])

# Generate holiday features
is_holiday = int(
    forecast_date in holidays["date"].values
)

is_national_holiday = int(
    forecast_date in holidays.loc[
        holidays["locale"] == "National", "date"
    ].values
)

is_regional_holiday = int(
    forecast_date in holidays.loc[
        holidays["locale"] == "Regional", "date"
    ].values
)

is_local_holiday = int(
    forecast_date in holidays.loc[
        holidays["locale"] == "Local", "date"
    ].values
)

# Retrieve oil price
oil_for_date = oil[
    oil["date"] == forecast_date
]

if not oil_for_date.empty:
    dcoilwtico = oil_for_date["dcoilwtico"].iloc[0]
else:
    previous_oil = oil[
        oil["date"] < forecast_date
    ].sort_values("date")

    if not previous_oil.empty:
        dcoilwtico = previous_oil["dcoilwtico"].iloc[-1]
    else:
        dcoilwtico = oil["dcoilwtico"].median()

# Retrieve historical sales
selected_history = sales_history[
    (sales_history["store_nbr"] == store_number) &
    (sales_history["family"] == product_family) &
    (sales_history["date"] < forecast_date)
].sort_values("date")

# Check that sufficient historical data exists
if len(selected_history) < 14:

    st.error(
        "There is not enough historical sales data to calculate "
        "the required lag features."
    )

else:

    # Calculate lag features
    lag_1 = selected_history["sales"].iloc[-1]
    lag_7 = selected_history["sales"].iloc[-7]
    lag_14 = selected_history["sales"].iloc[-14]

    rolling_mean_7 = selected_history["sales"].iloc[-7:].mean()
    rolling_mean_14 = selected_history["sales"].iloc[-14:].mean()

    # Create prediction input
    prediction_input = pd.DataFrame({
        "store_nbr": [store_number],
        "family": [product_family],
        "onpromotion": [onpromotion],
        "dcoilwtico": [dcoilwtico],
        "is_holiday": [is_holiday],
        "is_national_holiday": [is_national_holiday],
        "is_regional_holiday": [is_regional_holiday],
        "is_local_holiday": [is_local_holiday],
        "year": [year],
        "month": [month],
        "day": [day],
        "day_of_week_num": [day_of_week_num],
        "is_weekend": [is_weekend],
        "type": [store_type],
        "cluster": [cluster],
        "lag_1": [lag_1],
        "lag_7": [lag_7],
        "lag_14": [lag_14],
        "rolling_mean_7": [rolling_mean_7],
        "rolling_mean_14": [rolling_mean_14]
    })

    # Convert categorical variables into dummy variables
    prediction_input = pd.get_dummies(
        prediction_input,
        columns=["family", "type"],
        drop_first=True
    )

    # Match the model's training features
    prediction_input = prediction_input.reindex(
        columns=model_columns,
        fill_value=0
    )

    # Convert features to numeric format
    prediction_input = prediction_input.astype(float)

    # Prediction button
    st.subheader("Sales Prediction")

    if st.button("Predict Sales"):

        prediction_log = model.predict(prediction_input)

        # Convert prediction back from logarithmic scale
        prediction = np.expm1(prediction_log)

        # Prevent negative predictions
        prediction = max(float(prediction[0]), 0)

        st.metric(
            label="Estimated Sales",
            value=f"{prediction:,.2f}"
        )

        st.info(
            "This is an estimated sales value generated by "
            "the trained forecasting model."
        )
