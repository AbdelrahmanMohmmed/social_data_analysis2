"""Streamlit app with API prediction and local multi-model voting."""

import json
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st


st.set_page_config(page_title="Sentiment Predictor", page_icon="S", layout="wide")

BASE_DIR = Path(__file__).resolve().parent
SECTION4_DIR = BASE_DIR.parent / "section4"
SECTION4_MODELS_DIR = SECTION4_DIR / "models"
SECTION4_MODEL_FEATURES_DIR = SECTION4_DIR / "model"
SECTION4_LEX_DIR = SECTION4_DIR / "lex_model"
OPT_DIR = BASE_DIR / "benchmark" / "optimization"
AFINN_PATH = SECTION4_DIR / "AFINN" / "AFINN-111.txt"

NEGATION_WORDS = {
	"not", "no", "never", "neither", "nobody", "nothing", "none",
	"dont", "don't", "doesnt", "doesn't", "didnt", "didn't",
	"hasnt", "hasn't", "havent", "haven't", "cannot", "cant", "can't"
}


@st.cache_resource
def load_pickle(path_str):
	with Path(path_str).open("rb") as f:
		return pickle.load(f)


@st.cache_resource
def load_glove_model():
	import gensim.downloader as api

	return api.load("glove-wiki-gigaword-100")


@st.cache_resource
def load_vader_analyzer():
	import nltk
	from nltk.sentiment import SentimentIntensityAnalyzer

	try:
		nltk.data.find("vader_lexicon")
	except LookupError:
		nltk.download("vader_lexicon")

	return SentimentIntensityAnalyzer()


@st.cache_resource
def load_afinn_dict(path_str):
	d = {}
	p = Path(path_str)
	if not p.exists():
		return d
	with p.open("r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			w, s = line.split("\t")
			d[w.lower()] = int(s)
	return d


def discover_models():
	models = []

	# Section4 trained ML models
	for run_dir in sorted(SECTION4_MODELS_DIR.glob("clean_*")):
		cfg_path = run_dir / "training_config.json"
		enc_path = run_dir / "label_encoder.pkl"
		if not cfg_path.exists() or not enc_path.exists():
			continue

		with cfg_path.open("r", encoding="utf-8") as f:
			cfg = json.load(f)

		feature_file = Path(cfg.get("feature_file", ""))
		feature_type = cfg.get("feature_type", "unknown")
		tfidf_vectorizer = None
		if feature_type == "tfidf" and feature_file.name.endswith("_tfidf_matrix.csv"):
			vec_name = feature_file.name.replace("_tfidf_matrix.csv", "_tfidf_vectorizer.pkl")
			tfidf_vectorizer = SECTION4_MODEL_FEATURES_DIR / vec_name

		model_files = {
			"svm": run_dir / "svm_model.pkl",
			"logistic": run_dir / "logistic_regression_model.pkl",
			"random_forest": run_dir / "random_forest_model.pkl",
		}
		for model_key, model_path in model_files.items():
			if not model_path.exists():
				continue
			models.append(
				{
					"id": f"ml::{run_dir.name}::{model_key}",
					"display": f"ML | {run_dir.name} | {model_key}",
					"source": "ml",
					"model_path": str(model_path),
					"encoder_path": str(enc_path),
					"feature_type": feature_type,
					"tfidf_vectorizer": str(tfidf_vectorizer) if tfidf_vectorizer else None,
				}
			)

	# Lexical models
	models.append(
		{
			"id": "lex::vader",
			"display": "Lexical | VADER",
			"source": "lex",
			"model_key": "vader",
		}
	)
	models.append(
		{
			"id": "lex::afinn",
			"display": "Lexical | AFINN (negation)",
			"source": "lex",
			"model_key": "afinn",
		}
	)

	# Optimized models
	opt_encoder = OPT_DIR / "label_encoder.pkl"
	if opt_encoder.exists():
		for key, filename in {
			"svm": "svm_best_model.pkl",
			"logistic": "logistic_best_model.pkl",
			"random_forest": "random_forest_best_model.pkl",
		}.items():
			mp = OPT_DIR / filename
			if mp.exists():
				models.append(
					{
						"id": f"opt::{key}",
						"display": f"Optimized | {key}",
						"source": "optimized",
						"model_path": str(mp),
						"encoder_path": str(opt_encoder),
						"feature_type": "glove",
						"tfidf_vectorizer": None,
					}
				)

	return models


def text_to_glove(text):
	glove = load_glove_model()
	words = str(text).lower().split()
	vectors = [glove[w] for w in words if w in glove.key_to_index]
	if not vectors:
		return np.zeros(100, dtype=np.float32).reshape(1, -1)
	return np.mean(vectors, axis=0).astype(np.float32).reshape(1, -1)


def text_to_tfidf(text, vectorizer_path):
	vec = load_pickle(vectorizer_path)
	return vec.transform([text]).toarray()


def lexical_vader_predict(text):
	vader = load_vader_analyzer()
	compound = vader.polarity_scores(str(text))["compound"]
	if compound >= 0.05:
		return "Positive", float((compound + 1) / 2)
	if compound <= -0.05:
		return "Negative", float((abs(compound) + 1) / 2)
	return "Neutral", float(1 - abs(compound))


def tokenize_for_afinn(text):
	return re.findall(r"[a-zA-Z']+|[.!?]", str(text).lower())


def lexical_afinn_predict(text):
	afinn = load_afinn_dict(str(AFINN_PATH))
	if not afinn:
		return "Neutral", 0.0

	tokens = tokenize_for_afinn(text)
	score = 0
	scope = 0
	for tok in tokens:
		if tok in {".", "!", "?"}:
			scope = 0
			continue
		cleaned = tok.strip("'")
		if cleaned in NEGATION_WORDS:
			scope = 3
			continue
		if cleaned in afinn:
			ws = afinn[cleaned]
			if scope > 0:
				ws *= -1
			score += ws
		if scope > 0:
			scope -= 1

	if score > 0:
		sentiment = "Positive"
	elif score < 0:
		sentiment = "Negative"
	else:
		sentiment = "Neutral"

	confidence = min(abs(score) / 8.0, 1.0)
	return sentiment, float(confidence)


def predict_with_entry(entry, text):
	if entry["source"] == "lex":
		if entry["model_key"] == "vader":
			s, c = lexical_vader_predict(text)
		else:
			s, c = lexical_afinn_predict(text)
		return {"model": entry["display"], "sentiment": s, "confidence": c}

	model = load_pickle(entry["model_path"])
	encoder = load_pickle(entry["encoder_path"])

	if entry["feature_type"] == "tfidf":
		if not entry.get("tfidf_vectorizer"):
			raise ValueError(f"Missing TF-IDF vectorizer for {entry['display']}")
		x = text_to_tfidf(text, entry["tfidf_vectorizer"])
	else:
		x = text_to_glove(text)

	pred_enc = model.predict(x)[0]
	sentiment = encoder.inverse_transform([pred_enc])[0]

	if hasattr(model, "predict_proba"):
		conf = float(np.max(model.predict_proba(x)[0]))
	elif hasattr(model, "decision_function"):
		scores = np.array(model.decision_function(x)).reshape(-1)
		shifted = scores - np.max(scores)
		probs = np.exp(shifted) / np.sum(np.exp(shifted))
		conf = float(np.max(probs))
	else:
		conf = 1.0

	return {"model": entry["display"], "sentiment": str(sentiment), "confidence": conf}


def majority_vote(predictions):
	sentiments = [p["sentiment"] for p in predictions]
	counts = Counter(sentiments)
	top_count = max(counts.values())
	winners = [s for s, c in counts.items() if c == top_count]

	if len(winners) == 1:
		return winners[0], counts

	# Tie-break: highest mean confidence among tied sentiments
	best = None
	best_conf = -1.0
	for sentiment in winners:
		confs = [p["confidence"] for p in predictions if p["sentiment"] == sentiment]
		avg_conf = float(np.mean(confs)) if confs else 0.0
		if avg_conf > best_conf:
			best_conf = avg_conf
			best = sentiment
	return best, counts


st.title("Sentiment Prediction Dashboard")
tab_api, tab_voting = st.tabs(["API /predict", "Voting Ensemble"])

with tab_api:
	st.caption("Use FastAPI POST /predict backed by best optimized model")
	api_url = st.text_input("API endpoint", value="http://localhost:8000/predict", key="api_url")
	api_text = st.text_area("Input text", height=120, key="api_text")

	if st.button("Predict via API", type="primary"):
		if not api_text.strip():
			st.warning("Please enter text before prediction.")
		else:
			try:
				resp = requests.post(api_url, json={"text": api_text.strip()}, timeout=20)
				resp.raise_for_status()
				data = resp.json()
				st.success("Prediction complete")
				st.write(f"Sentiment: **{data.get('sentiment', 'Unknown')}**")
				st.write(f"Confidence: **{float(data.get('confidence', 0.0)):.2%}**")
				st.write(f"Model: **{data.get('model', 'Unknown')}**")
			except requests.exceptions.RequestException as exc:
				st.error(f"Request failed: {exc}")
			except Exception as exc:
				st.error(f"Unexpected error: {exc}")

	st.caption("Run API: uvicorn fastapi_predict:app --reload --port 8000")

with tab_voting:
	st.caption("Load models from section4 + lexical + optimized and vote on sentiment")
	all_models = discover_models()

	if not all_models:
		st.error("No models discovered. Check section4/models and benchmark/optimization artifacts.")
	else:
		model_labels = [m["display"] for m in all_models]
		default_selection = model_labels

		selected_labels = st.multiselect(
			"Choose models for voting",
			options=model_labels,
			default=default_selection,
		)

		voting_text = st.text_area("Input text for voting", height=140, key="voting_text")

		if st.button("Run Voting", type="primary"):
			if not voting_text.strip():
				st.warning("Please enter text before prediction.")
			elif not selected_labels:
				st.warning("Select at least one model.")
			else:
				selected_entries = [m for m in all_models if m["display"] in selected_labels]
				predictions = []
				errors = []

				with st.spinner("Running selected models..."):
					for entry in selected_entries:
						try:
							predictions.append(predict_with_entry(entry, voting_text.strip()))
						except Exception as exc:
							errors.append(f"{entry['display']}: {exc}")

				if predictions:
					voted_sentiment, vote_counts = majority_vote(predictions)
					avg_conf = float(np.mean([p["confidence"] for p in predictions]))

					st.success("Voting complete")
					st.subheader("Final Voting Result")
					st.write(f"Sentiment: **{voted_sentiment}**")
					st.write(f"Average confidence: **{avg_conf:.2%}**")
					st.write(f"Vote counts: **{dict(vote_counts)}**")

					st.subheader("Per-model Predictions")
					df = pd.DataFrame(predictions)
					df["confidence"] = df["confidence"].map(lambda x: f"{x:.2%}")
					st.dataframe(df, use_container_width=True)

				if errors:
					st.subheader("Model Errors")
					for err in errors:
						st.error(err)
