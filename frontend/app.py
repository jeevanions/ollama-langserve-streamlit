import streamlit as st
import requests
import os
st.title("AI Bill Of Rights")
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
# Sample input form
input_value = st.text_input("Enter some input:")
if st.button("Submit"):
    response = requests.post(f"{backend_url}/query_with_timing", 
                             headers={"Content-Type": "application/json"},
                             json={"query": input_value})
    if response.status_code == 200:
        data = response.json()
        st.write("Backend response:", data["response"])
        st.write("Response time:", data["response_time"], "seconds")
    else:
        st.write("Error:", response.status_code)