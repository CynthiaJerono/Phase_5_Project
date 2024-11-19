from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sqlite3
from datetime import datetime
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification

# Initialize FastAPI app
app = FastAPI()

# Load DistilBERT tokenizer and model pipeline
try:
    model_dir = "C:\\Users\\PC\\Music\\DSC_PHASE_V\\CYNTHIA_PHASE_V\\Phase_5_Project\\distilbert_model_pipeline"
    classifier = pipeline(
        "text-classification",
        model=DistilBertForSequenceClassification.from_pretrained(
            model_dir, trust_remote_code=False
        ),
        tokenizer=DistilBertTokenizerFast.from_pretrained(model_dir),
        top_k=None  # Replace deprecated return_all_scores
    )
    print("DistilBERT pipeline loaded successfully.")
except Exception as e:
    print(f"Error loading DistilBERT pipeline: {e}")
    raise RuntimeError("Failed to load the DistilBERT pipeline. Ensure the model directory exists and is correctly configured.")

# SQLite Database Connection
def connect_db():
    conn = sqlite3.connect("mental_health.db")
    return conn

# Create a table for history if it doesn't exist
def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            text TEXT,
            label TEXT,
            timestamp DATETIME
        )
    """)
    conn.commit()
    conn.close()

create_table()

# Pydantic model for input
class TextInput(BaseModel):
    text: str
    user_id: int

# Label mapping
label_map = {
    "LABEL_0": "non_mental_health_issue",
    "LABEL_1": "mental_health_issue"
}

# API endpoint to predict sentiment
@app.post("/predict")
async def predict(input_data: TextInput):
    try:
        # Use the DistilBERT pipeline for prediction
        prediction = classifier(input_data.text)
        if not prediction or not isinstance(prediction, list):
            raise ValueError("Invalid prediction output.")

        # Extract label with the highest score
        highest_score_prediction = max(prediction[0], key=lambda x: x["score"])
        raw_label = highest_score_prediction["label"]
        predicted_label = label_map.get(raw_label, "Unknown")

        # Save prediction to history
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO history (user_id, text, label, timestamp)
            VALUES (?, ?, ?, ?)
        """, (input_data.user_id, input_data.text, predicted_label, datetime.now()))
        conn.commit()
        conn.close()

        # Return response
        return {
            "user_id": input_data.user_id,
            "text": input_data.text,
            "label": predicted_label,
            "confidence_score": highest_score_prediction["score"]
        }
    except Exception as e:
        print(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Error during prediction.")

# API endpoint to retrieve user history
@app.get("/history/{user_id}")
async def get_history(user_id: int):
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM history WHERE user_id=?", (user_id,))
        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {"message": f"No history found for user ID {user_id}."}

        history = [{"id": row[0], "user_id": row[1], "text": row[2], "label": row[3], "timestamp": row[4]} for row in rows]
        return {"history": history}
    except Exception as e:
        print(f"Error retrieving history: {e}")
        raise HTTPException(status_code=500, detail="Error retrieving user history.")

# API endpoint to clear the history (optional)
@app.delete("/history/clear")
async def clear_history():
    try:
        conn = connect_db()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM history")
        conn.commit()
        conn.close()
        return {"message": "All history has been cleared."}
    except Exception as e:
        print(f"Error clearing history: {e}")
        raise HTTPException(status_code=500, detail="Error clearing history.")









