import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

# Load the dataset from CSV file
file_path = '1.csv'  # Replace with the actual path to your CSV file
df = pd.read_csv(file_path)

# Train a simple Decision Tree model
X = df.drop('label', axis=1)
y = df['label']

model = DecisionTreeClassifier()
model.fit(X, y)

# Function to predict crop label
def predict_crop_label(user_inputs):
    prediction = model.predict(pd.DataFrame([user_inputs]))
    return prediction[0]

# Sidebar navigation
page = st.sidebar.selectbox("Select a page", ["Home", "Crop Prediction"])

# Main content
if page == "Home":
    st.title("Welcome to the Crop Prediction App")
    st.write("This app helps you predict the crop label based on input features.")
    st.write("")

elif page == "Crop Prediction":
    st.title('Crop Prediction Dashboard')

    # User input for prediction
    st.sidebar.header('User Input Features')

    # Collect user inputs
    user_inputs = {
        'N': st.sidebar.slider('Nitrogen (N)', min_value=0, max_value=100, value=50),
        'P': st.sidebar.slider('Phosphorus (P)', min_value=0, max_value=100, value=50),
        'K': st.sidebar.slider('Potassium (K)', min_value=0, max_value=100, value=50),
        'temperature': st.sidebar.slider('Temperature', min_value=0.0, max_value=40.0, value=25.0),
        'humidity': st.sidebar.slider('Humidity', min_value=0.0, max_value=100.0, value=50.0),
        'ph': st.sidebar.slider('pH', min_value=0.0, max_value=14.0, value=7.0),
        'rainfall': st.sidebar.slider('Rainfall', min_value=0.0, max_value=300.0, value=150.0),
    }

    # Display the user inputs
    st.sidebar.write('## User Input Features')
    st.sidebar.write(pd.DataFrame(user_inputs, index=[0]))

    # Predict the label
    prediction = predict_crop_label(user_inputs)
    st.write('## Prediction')
    st.write(f'The predicted crop label is: {prediction}')
