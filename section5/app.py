"""Streamlit UI for FastAPI sentiment prediction endpoint."""

import requests
import streamlit as st


st.set_page_config(page_title="Sentiment Predictor", page_icon="S", layout="centered")

st.title("Sentiment Prediction")
st.caption("Uses FastAPI POST /predict backed by the best optimized model")

default_api_url = "http://localhost:8000/predict"
api_url = st.text_input("API endpoint", value=default_api_url)

text = st.text_area("Enter text", height=140, placeholder="Type a review or comment...")

if st.button("Predict", type="primary"):
	if not text.strip():
		st.warning("Please enter text before prediction.")
	else:
		try:
			response = requests.post(
				api_url,
				json={"text": text.strip()},
				timeout=20,
			)
			response.raise_for_status()
			data = response.json()

			sentiment = data.get("sentiment", "Unknown")
			confidence = float(data.get("confidence", 0.0))
			model_name = data.get("model", "Unknown")

			st.success("Prediction complete")
			st.write(f"Sentiment: **{sentiment}**")
			st.write(f"Confidence: **{confidence:.2%}**")
			st.write(f"Model: **{model_name}**")
			st.progress(min(max(confidence, 0.0), 1.0))

		except requests.exceptions.RequestException as exc:
			st.error(f"Request failed: {exc}")
		except Exception as exc:
			st.error(f"Unexpected error: {exc}")

st.markdown("---")
st.caption("Run API: uvicorn fastapi_predict:app --reload --port 8000")
