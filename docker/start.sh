#!/bin/bash

# Start the FastAPI server
uvicorn unified_app:app --host 0.0.0.0 --port 8000 &

# Start the Streamlit app
streamlit run unified_app.py --server.port 8501 --server.address 0.0.0.0