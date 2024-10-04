import streamlit as st
import pandas as pd
import json
import requests

# Function to make a POST request to the FastAPI endpoint
def call_fastapi_model(file):
    # Send the CSV file to the FastAPI endpoint using a POST request
    url = 'http://127.0.0.1:8000/run-model/mlm'
    
    # Prepare the file to be sent
    files = {'file': (file.name, file.getvalue(), 'text/csv')}
    
    try:
        response = requests.post(url, files=files)
        
        # Check if the request was successful
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"Failed to call API: {response.status_code}"}
    
    except Exception as e:
        return {"error": f"An error occurred: {str(e)}"}

# Streamlit UI
st.title("CSV to FastAPI Model Processor")

# Upload CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Display file name
    st.write(f"Uploaded file: {uploaded_file.name}")
    
    # Call FastAPI with the uploaded file
    result = call_fastapi_model(uploaded_file)
    
    # Display the result from the API
    st.write("Response from API:")
    st.json(result)
