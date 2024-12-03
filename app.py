import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import joblib

# Custom CSS for background styling
page_bg_css = """
<style>
body {
    background-color: #5B3A6A; /* Example color resembling the image */
    color: white;
}
h1, h2, h3, h4, h5, h6 {
    color: white;
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# Streamlit app title
st.title('''A Data-Driven Solution for Holy Cross College''')

# Uploading the dataset
uploaded_file = st.file_uploader("Upload your housing dataset (CSV file)", type="csv")

if uploaded_file is not None:
    # Loading the dataset
    housing_data = pd.read_csv(uploaded_file)
    st.write("Initial Data Info:")
    st.write(housing_data.info())

    st.write("Sample Data:")
    st.write(housing_data.head())

    # Ensuring 'Hall' column exists
    target_column = 'Hall'
    if target_column not in housing_data.columns:
        st.error(f"Target column '{target_column}' not found in the dataset.")
    else:
        # Dropping rows with 'UEDGE' in 'Hall' column if it exists
        housing_data = housing_data[housing_data[target_column] != 'UEDGE']
        housing_data[target_column] = housing_data[target_column].astype('category')

        # Dropping unnecessary or unnamed columns
        unnamed_cols = [col for col in housing_data.columns if "Unnamed" in col or housing_data[col].isnull().all()]
        housing_data.drop(columns=unnamed_cols, inplace=True)

        # Handling missing values
        housing_data.dropna(inplace=True)

        # Separating features and target variable
        X = housing_data.drop(columns=[target_column], errors='ignore')
        y = housing_data[target_column]

        # Handling categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Splitting data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Training the Random Forest model
        rf_model = RandomForestClassifier(n_estimators=1000, max_features='sqrt', random_state=42)
        rf_model.fit(X_train, y_train)

        # Saving the model
        model_path = 'random_forest_model.pkl'
        joblib.dump(rf_model, model_path)
        st.success(f"Random Forest model trained and saved successfully at {model_path}!")

        # Making predictions
        st.header("Make Predictions")
        uploaded_test_file = st.file_uploader("Upload a dataset for predictions (CSV file)", type="csv")
        if uploaded_test_file is not None:
            test_data = pd.read_csv(uploaded_test_file)

            # Ensuring compatibility with training data
            test_data = pd.get_dummies(test_data, drop_first=True)
            missing_cols = set(X.columns) - set(test_data.columns)
            for col in missing_cols:
                test_data[col] = 0
            test_data = test_data[X.columns]

            # Making predictions
            predictions = rf_model.predict(test_data)
            test_data[target_column] = predictions

            st.write("Predictions:")
            st.write(test_data.head())

            # Download link for predictions
            csv = test_data.to_csv(index=False)
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name="predictions.csv",
                mime="text/csv"
            )

        # Evaluating the model
        preds = rf_model.predict(X_test)

        st.write("Confusion Matrix:")
        st.write(confusion_matrix(y_test, preds))

        st.write("Classification Report:")
        st.text(classification_report(y_test, preds))

        # Download link for the model
        with open(model_path, "rb") as file:
            st.download_button(
                label="Download Trained Model",
                data=file,
                file_name="random_forest_model.pkl",
                mime="application/octet-stream"
            )
