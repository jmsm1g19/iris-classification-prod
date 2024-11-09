import streamlit as st
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import network_objects as no
import joblib
import nest_asyncio
import uvicorn
from threading import Thread

# Enable nested async loops (needed for running FastAPI alongside Streamlit)
nest_asyncio.apply()

# Common initialization code
parameters_path = "data/best_params.pkl"
loaded_parameters = joblib.load(parameters_path)

scaler_path = "data/scaler.pkl"
scaler = joblib.load(scaler_path)

petals = {
    0: "I. setosa",
    1: "I. versicolor",
    2: "I. virginica"
}

# Shared prediction function
def predict_species(sepal_len, sepal_wid, petal_len, petal_wid):
    input_data = np.array([[sepal_len, sepal_wid, petal_len, petal_wid]])
    input_scaled = scaler.transform(input_data)
    prediction = no.predict(input_scaled.T, loaded_parameters)
    return petals[int(prediction)]

# FastAPI setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    sepal_len: float
    sepal_wid: float
    petal_len: float
    petal_wid: float

@app.post("/which_species_is_this")
async def api_predict_species(item: Item):
    return {"prediction": predict_species(item.sepal_len, item.sepal_wid, 
                                       item.petal_len, item.petal_wid)}

# Streamlit UI
st.title('Iris Species Classifier')
st.write('Enter the measurements below to classify the Iris species.')

# API documentation
with st.expander("API Documentation"):
    st.markdown("""
    ### API Endpoint
    POST `/which_species_is_this`
    
    ### Example Request
    ```bash
    curl -X POST "http://localhost:8000/which_species_is_this" \\
         -H "Content-Type: application/json" \\
         -d '{"sepal_len": 5.1, "sepal_wid": 3.5, "petal_len": 1.4, "petal_wid": 0.2}'
    ```
    """)

col1, col2 = st.columns(2)
with col1:
    sepal_length = st.slider('Sepal Length (cm)', 4.0, 8.0, 5.0, 0.1)
    sepal_width = st.slider('Sepal Width (cm)', 2.0, 4.5, 3.0, 0.1)
with col2:
    petal_length = st.slider('Petal Length (cm)', 1.0, 7.0, 4.0, 0.1)
    petal_width = st.slider('Petal Width (cm)', 0.1, 2.5, 1.0, 0.1)

if st.button('Predict Species'):
    species = predict_species(sepal_length, sepal_width, petal_length, petal_width)
    st.success(f'Predicted Species: {species}')
    
    measurements_df = {
        'Measurement': ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'],
        'Value (cm)': [sepal_length, sepal_width, petal_length, petal_width]
    }
    st.table(measurements_df)
