import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# Ensure matplotlib uses the correct backend
matplotlib.use("Agg")

# Base URL for the FastAPI server
API_BASE_URL = "http://127.0.0.1:8000"

# Streamlit app title and description
st.title("Mental Health Detection via Online Textual Conversations")
st.write("Enter text to predict mental health labels")

# Input field for user ID
user_id = st.text_input("Enter your User ID (numeric)")

# Validate that user ID is numeric
if user_id and not user_id.isdigit():
    st.error("User ID must be a numeric value.")

# Section to display user history and mental health trend if a valid user ID is provided
if user_id and user_id.isdigit():
    try:
        st.write("Fetching user history...")
        response = requests.get(f"{API_BASE_URL}/history/{user_id}")
        
        if response.status_code == 200:
            data = response.json()
            if "history" in data and data["history"]:  # Check if 'history' exists and is not empty
                # Create a DataFrame from the history data
                history_df = pd.DataFrame(data["history"])
                
                # Format timestamp for readability
                history_df["timestamp"] = pd.to_datetime(history_df["timestamp"]).dt.strftime("%Y-%m-%d %H:%M:%S")
                st.write("User Prediction History:")
                st.dataframe(history_df)

                # Plot mental health label trends over time
                plt.figure()  # Clear the plot
                label_counts = history_df["label"].value_counts()
                label_counts.plot(kind="bar", title="Mental Health Labels Frequency")
                plt.xlabel("Label")
                plt.ylabel("Frequency")
                st.pyplot(plt)
            else:
                st.write("No prediction history available for this user.")
        else:
            st.error(f"Error {response.status_code}: Could not retrieve user history.")
    except Exception as e:
        st.error(f"Error: {e}")

# Text area for user input
text = st.text_area("Enter text here")

# Button to submit the text for prediction
if st.button("Predict"):
    if text and user_id and user_id.isdigit():
        try:
            st.write("Submitting prediction request...")
            response = requests.post(
                f"{API_BASE_URL}/predict", 
                json={"text": text, "user_id": int(user_id)}
            )
            
            if response.status_code == 200:
                label = response.json().get("label")
                st.write(f"Predicted Label: **{label}**")

                # Add resource suggestions based on the prediction
                if label == "Depression":
                    st.info("We recommend you visit a counselor or take a mental health screening test.")
                elif label == "Anxiety":
                    st.info("Consider practicing mindfulness or relaxation techniques. Seek professional help if needed.")
                else:
                    st.success("It's always good to talk about your thoughts with someone.")
            else:
                st.error(f"Error {response.status_code}: Unable to get a prediction.")
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Please enter both text and a valid numeric User ID to make a prediction.")
