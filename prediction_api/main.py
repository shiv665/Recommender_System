from fastapi import FastAPI, HTTPException
import json
import os

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="Netflix RecSys Prediction API", 
    description="API for serving precomputed Netflix recommendations", 
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load recommendations on startup
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "recommendations.json")
recommendations = {}

@app.on_event("startup")
def load_data():
    global recommendations
    try:
        with open(DATA_PATH, "r") as f:
            recommendations = json.load(f)
        print(f"Loaded {len(recommendations)} users' recommendations.")
    except Exception as e:
        print(f"Failed to load recommendations: {e}")

@app.get("/")
def root():
    return {
        "message": "Netflix RecSys API is running.",
        "endpoints": [
            "/docs (Swagger UI)",
            "/users (List all valid user IDs)",
            "/predict/{user_id} (Get predictions for a user)"
        ]
    }

@app.get("/users")
def get_users():
    """Return a list of available user IDs for testing."""
    return {"user_ids": list(recommendations.keys())}

@app.get("/predict/{user_id}")
def get_predictions(user_id: str):
    """Get recommendations for a specific user."""
    if user_id not in recommendations:
        raise HTTPException(status_code=404, detail="User not found or no predictions available.")
    return recommendations[user_id]
