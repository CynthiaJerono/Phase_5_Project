from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import joblib
import pandas as pd

# Load your model
model = joblib.load("random_forest_pipeline_model.pkl")

# Initialize FastAPI app
app = FastAPI()

# Define a structure for the input data
class RedditPost(BaseModel):
    title: Optional[str] = None
    subreddit: Optional[str] = None
    post_score: Optional[int] = 0
    comment_score: Optional[int] = 0
    upvote_ratio: Optional[float] = 0.5
    total_awards_received: Optional[int] = 0
    post_length: Optional[int] = 0
    post_flair: Optional[str] = None
    over_18: Optional[bool] = False
    sentiment: Optional[float] = 0.0
    cleaned_post_body: Optional[str] = ""
    cleaned_comment_body: Optional[str] = ""
    post_num_comments: Optional[int] = 0

@app.post("/predict/")
def predict(posts: List[RedditPost]):
    # Convert input data to DataFrame
    data = pd.DataFrame([post.dict() for post in posts])
    data.fillna({
        'post_score': 0,
        'comment_score': 0,
        'upvote_ratio': 0,
        'total_awards_received': 0,
        'post_length': 0,
        'sentiment': 0.0,
        'post_num_comments': 0,
    }, inplace=True)

    # Run predictions
    try:
        predictions = model.predict(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # Map predictions
    label_map = {0: "happy", 1: "mental health issue", 2: "neutral"}
    result = [label_map[pred] for pred in predictions]

    return {"predictions": result}


