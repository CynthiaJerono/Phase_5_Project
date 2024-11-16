from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from pydantic import BaseModel
import pickle
import sqlite3
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS middleware to allow requests from frontend (if hosted separately)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (use specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint for the API
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Mental Health Prediction API"}

# Load the Random Forest model
try:
    with open("random_forest_pipeline_model.pkl", "rb") as file:
        random_forest_pipeline_model = pickle.load(file)
except FileNotFoundError:
    raise RuntimeError("Model file 'random_forest_pipeline_model.pkl' not found. Please ensure the file exists.")

# SQLite Database Connection
def connect_db():
    conn = sqlite3.connect("mental_health.db")
    conn.row_factory = sqlite3.Row  # Return rows as dictionaries
    return conn

# Create the history table if it doesn't exist
def create_table():
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            text TEXT NOT NULL,
            label TEXT NOT NULL,
            timestamp DATETIME NOT NULL
        );
    """)
    conn.commit()
    conn.close()

# Ensure the history table is created on startup
create_table()

# Define a Pydantic model for input validation
class TextInput(BaseModel):
    text: str
    user_id: int

# WebSocket endpoint for real-time predictions
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive text data from WebSocket
            data = await websocket.receive_text()

            # Use Random Forest model to predict
            prediction = random_forest_pipeline_model.predict([data])[0]
            labels = {0: "Neutral", 1: "Depression", 2: "Anxiety"}
            label = labels.get(prediction, "Unknown")

            # Send prediction back to the client
            await websocket.send_text(f"Prediction: {label}")
    except WebSocketDisconnect:
        print("Client disconnected from WebSocket.")

# HTTP GET endpoint to retrieve user history by user_id
@app.get("/history/{user_id}")
async def get_history(user_id: int):
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM history WHERE user_id=?", (user_id,))
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        raise HTTPException(status_code=404, detail="No history found for this user")

    # Convert rows to a dictionary format for JSON response
    history = [
        {
            "id": row["id"],
            "text": row["text"],
            "label": row["label"],
            "timestamp": row["timestamp"],
        }
        for row in rows
    ]
    return {"history": history}

# HTTP POST endpoint for making predictions and saving them to the database
@app.post("/predict")
async def predict(input_data: TextInput):
    # Make prediction using Random Forest model
    prediction = random_forest_pipeline_model.predict([input_data.text])[0]
    labels = {0: "Neutral", 1: "Depression", 2: "Anxiety"}
    label = labels.get(prediction, "Unknown")

    # Insert prediction and text into the history database
    conn = connect_db()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO history (user_id, text, label, timestamp)
        VALUES (?, ?, ?, ?)
    """, (input_data.user_id, input_data.text, label, datetime.now()))
    conn.commit()
    conn.close()

    # Return the prediction as a JSON response
    return {
        "user_id": input_data.user_id,
        "text": input_data.text,
        "label": label,
    }
