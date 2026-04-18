"""FastAPI inference service for the best optimized sentiment model.

Provides one endpoint:
POST /predict
"""

import json
import pickle
from pathlib import Path
from typing import Optional

import gensim.downloader as api
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
OPT_DIR = BASE_DIR / "benchmark" / "optimization"


class PredictRequest(BaseModel):
	text: str = Field(..., min_length=1, description="Input text to classify")


class PredictResponse(BaseModel):
	sentiment: str
	confidence: float
	model: str


class InferenceService:
	def __init__(self):
		self.model = None
		self.label_encoder = None
		self.model_name = None
		self.glove_model = None
		self.expected_dim = 100

	def load(self):
		summary_path = OPT_DIR / "optimization_summary.json"
		if not summary_path.exists():
			raise FileNotFoundError(
				f"Missing optimization summary: {summary_path}. Run optimize_models.py first."
			)

		with summary_path.open("r", encoding="utf-8") as f:
			summary = json.load(f)

		self.model_name = summary["best_model"]["name"]
		features_file = str(summary.get("features_file", ""))

		model_file_map = {
			"svm": "svm_best_model.pkl",
			"logistic": "logistic_best_model.pkl",
			"random_forest": "random_forest_best_model.pkl",
		}
		if self.model_name not in model_file_map:
			raise ValueError(f"Unsupported best model: {self.model_name}")

		# Current inference server supports GloVe-based optimized models.
		if "glove" not in features_file.lower():
			raise ValueError(
				"Best optimized model was not trained on GloVe features. "
				"Run optimize_models.py with a GloVe features file for API inference."
			)

		model_path = OPT_DIR / model_file_map[self.model_name]
		encoder_path = OPT_DIR / "label_encoder.pkl"

		if not model_path.exists() or not encoder_path.exists():
			raise FileNotFoundError("Missing optimized model or label encoder artifacts")

		with model_path.open("rb") as f:
			self.model = pickle.load(f)
		with encoder_path.open("rb") as f:
			self.label_encoder = pickle.load(f)

		# Load cached GloVe model (downloads once, then uses local cache).
		self.glove_model = api.load("glove-wiki-gigaword-100")

	def text_to_glove_vector(self, text: str) -> np.ndarray:
		words = str(text).lower().split()
		vectors = [self.glove_model[w] for w in words if w in self.glove_model.key_to_index]
		if not vectors:
			return np.zeros(self.expected_dim, dtype=np.float32)
		return np.mean(vectors, axis=0).astype(np.float32)

	def predict(self, text: str) -> PredictResponse:
		if self.model is None or self.label_encoder is None or self.glove_model is None:
			raise RuntimeError("Model is not loaded")

		x = self.text_to_glove_vector(text).reshape(1, -1)
		encoded_pred = self.model.predict(x)[0]
		sentiment = self.label_encoder.inverse_transform([encoded_pred])[0]

		confidence: Optional[float] = None
		if hasattr(self.model, "predict_proba"):
			proba = self.model.predict_proba(x)[0]
			confidence = float(np.max(proba))
		elif hasattr(self.model, "decision_function"):
			scores = self.model.decision_function(x)
			# Fallback confidence from normalized margin-like score.
			scores = np.array(scores).reshape(-1)
			shifted = scores - np.max(scores)
			exp_scores = np.exp(shifted)
			confidence = float(np.max(exp_scores / np.sum(exp_scores)))
		else:
			confidence = 1.0

		return PredictResponse(
			sentiment=str(sentiment),
			confidence=confidence,
			model=self.model_name,
		)


service = InferenceService()
app = FastAPI(title="Sentiment API", version="1.0.0")


@app.on_event("startup")
def startup_event():
	service.load()


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
	try:
		return service.predict(payload.text)
	except Exception as exc:
		raise HTTPException(status_code=500, detail=str(exc))

