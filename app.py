import streamlit as st
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="centered"
)

st.title("❤️ Heart Disease Prediction")
st.write("Machine Learning demo based on the Heart Disease UCI dataset.")

st.info(
    "This demo is for educational purposes only. It is not a medical diagnosis "
    "and should not be used to make healthcare decisions."
)

@st.cache_resource
def load_model():
    # Load the same dataset used by the notebook
    dataset = pd.read_csv("dataset.csv")

    # Apply the same preprocessing used in the notebook
    dataset = pd.get_dummies(
        dataset,
        columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    )

    columns_to_scale = ["age", "trestbps", "chol", "thalach", "oldpeak"]
    scaler = StandardScaler()
    dataset[columns_to_scale] = scaler.fit_transform(dataset[columns_to_scale])

    X = dataset.drop(["target"], axis=1)
    y = dataset["target"]

    # KNN with 8 neighbors, matching the model selected in the notebook
    model = KNeighborsClassifier(n_neighbors=8)
    model.fit(X, y)

    return model, scaler, X.columns, columns_to_scale


model, scaler, feature_columns, columns_to_scale = load_model()

st.subheader("Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=55)
    sex = st.selectbox("Sex", [1, 0], format_func=lambda x: "Male (1)" if x == 1 else "Female (0)")
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        [0, 1, 2, 3],
        format_func=lambda x: f"Type {x}"
    )
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=130)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=50, max_value=700, value=240)
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl (fbs)",
        [0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )
    restecg = st.selectbox("Resting ECG Result (restecg)", [0, 1, 2])

with col2:
    thalach = st.number_input("Maximum Heart Rate (thalach)", min_value=50, max_value=250, value=150)
    exang = st.selectbox(
        "Exercise Induced Angina (exang)",
        [0, 1],
        format_func=lambda x: "No (0)" if x == 0 else "Yes (1)"
    )
    oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of Peak Exercise ST Segment (slope)", [0, 1, 2])
    ca = st.selectbox("Number of Major Vessels (ca)", [0, 1, 2, 3, 4])
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

if st.button("Predict", type="primary"):
    # Build one raw patient row using the same original feature names
    patient = pd.DataFrame([{
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }])

    # Apply the same categorical encoding
    patient = pd.get_dummies(
        patient,
        columns=["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
    )

    # Make the patient columns identical to the training columns
    patient = patient.reindex(columns=feature_columns, fill_value=0)

    # Apply the same scaling
    patient[columns_to_scale] = scaler.transform(patient[columns_to_scale])

    prediction = model.predict(patient)[0]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("Prediction: Heart disease is indicated by the model.")
    else:
        st.success("Prediction: Heart disease is not indicated by the model.")

    st.caption(
        "The result is a machine-learning prediction based on the supplied values, "
        "not a clinical diagnosis."
    )
