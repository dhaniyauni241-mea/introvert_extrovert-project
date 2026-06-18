from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI()

# Load trained model
model = joblib.load("model.joblib")


class PersonalityInput(BaseModel):
    Time_spent_Alone: float
    Stage_fear: int
    Social_event_attendance: float
    Going_outside: float
    Drained_after_socializing: int
    Friends_circle_size: float
    Post_frequency: float


@app.get("/")
def home():
    return {"message": "Introvert-Extrovert Prediction API Running"}


@app.post("/predict")
def predict(data: PersonalityInput):

    input_data = pd.DataFrame([{
        "Time_spent_Alone": data.Time_spent_Alone,
        "Stage_fear": data.Stage_fear,
        "Social_event_attendance": data.Social_event_attendance,
        "Going_outside": data.Going_outside,
        "Drained_after_socializing": data.Drained_after_socializing,
        "Friends_circle_size": data.Friends_circle_size,
        "Post_frequency": data.Post_frequency
    }])

    prediction = model.predict(input_data)[0]

    personality = (
        "Introvert"
        if str(prediction).lower() in ["introvert", "0"]
        else "Extrovert"
    )

    return {
        "prediction": personality
    }