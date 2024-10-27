import uvicorn
import logging
from fastapi import FastAPI, HTTPException, status

from src.module_6.src.basket_model.feature_store import FeatureStore
from src.module_6.src.basket_model.basket_model import BasketModel


class Features(str):
    user_id: str


logger = logging.getLogger(__name__)

# Configure the logger
logging.basicConfig(
    filename="logs.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

feature_store = FeatureStore()
model = BasketModel()

# Create an instance of FastAPI
app = FastAPI()


# Define a route for the root URL ("/")
@app.get("/")
def read_root():
    logging.info("Hello, World!")
    return {"message": "Hello, World!"}


@app.get("/status")
def check():
    logging.info("Status OK")
    return {"status": "OK"}


@app.post("/predict/{user_id}")
async def predict(user_id: str):
    try:
        user_features = feature_store.get_features(user_id)
        input_data = user_features.to_numpy()
        prediction = model.predict(input_data)
        logger.info(f"Prediction for user {user_id}: {prediction}")
        return {"price": prediction[0]}
    except Exception as e:
        logger.error(f"HTTP Error predicting for user {user_id}: {e}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))


# This block allows you to run the application using Uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
