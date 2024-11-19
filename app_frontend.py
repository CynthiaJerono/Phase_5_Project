import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Add a picture at the top of the app
st.image("image002.jpg", caption="Mental Health Awareness", use_container_width=True)

# App title and description
st.title("Mental Health Detection via Online Textual Conversations")
st.write(
    """
    This app allows users to predict whether a piece of text is related to mental health issues or not.
    """
)

# Section: Predict Sentiment from Text
st.header("Predict Mental Health Sentiment from Text")

# Input fields
user_id = st.text_input("Enter your User ID (integer):")
text_input = st.text_area("Enter your text for prediction:")

# Button for prediction
if st.button("Predict Sentiment"):
    if user_id.isdigit() and text_input.strip():
        try:
            response = requests.post(
                "http://127.0.0.1:8000/predict",
                json={"text": text_input, "user_id": int(user_id)},
            )
            if response.status_code == 200:
                result = response.json()
                predicted_label = result.get("label", "Unknown")
                confidence_score = result.get("confidence_score", None)

                # Display prediction result
                st.write(f"**Predicted Label:** {predicted_label}")
                if confidence_score is not None:
                    st.write(f"**Confidence Score:** {confidence_score:.2f}")

                # Display additional messages based on prediction
                if predicted_label == "non_mental_health_issue":
                    st.success("The text does not appear to indicate mental health issues. Keep up the positivity!")
                    st.write(
                        "Remember to stay connected with loved ones and continue practicing self-care. 😊"
                    )
                elif predicted_label == "mental_health_issue":
                    st.warning("The text indicates potential mental health concerns.")
                    st.write(
                        "Consider reaching out to a counselor or mental health professional for support. You are not alone. 💛"
                    )
                else:
                    st.info("The prediction result is unclear. Please try with more detailed text.")
            else:
                st.error("Error: Unable to connect to the backend. Please try again later.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid User ID and text.")

# Section: User Prediction History
st.header("User Prediction History")

# Button for viewing prediction history
if user_id.isdigit():
    try:
        response = requests.get(f"http://127.0.0.1:8000/history/{user_id}")
        if response.status_code == 200:
            history_data = response.json().get("history", [])
            if history_data:
                history_df = pd.DataFrame(history_data)
                st.write("Prediction History:")
                st.write(history_df)

                # Plot distribution of sentiments
                sentiment_counts = history_df["label"].value_counts()
                plt.figure(figsize=(8, 4))
                sentiment_counts.plot(kind="bar", color=["green", "red"], edgecolor="black")
                plt.title("Sentiment Distribution")
                plt.xlabel("Sentiment")
                plt.ylabel("Frequency")
                st.pyplot(plt)
            else:
                st.info("No prediction history found for this User ID.")
        else:
            st.error("Error: Unable to retrieve history. Please try again later.")
    except Exception as e:
        st.error(f"An error occurred while retrieving history: {e}")





















