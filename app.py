"""
app.py
Flask Web Application for Multilingual AI Tourist Assistant.
Exposes endpoints for:
- /api/chat (Intent classification + Semantic QA)
- /api/translate (Real-time online/offline translation)
- /api/recommend (Location & preference recommendations)
- /api/phrases (Offline survival phrasebook)
- /api/metrics (NLP evaluation results)
"""

import os
import sys
import json
from flask import Flask, render_template, request, jsonify

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.intent_classifier import TouristIntentClassifier
from src.semantic_retriever import DestinationRetriever
from src.translator import MultilingualTranslator, SUPPORTED_LANGUAGES
from src.recommender import TouristRecommender, CITY_COORDINATES

app = Flask(__name__, static_folder="static", template_folder="templates")

# Initialize models
print("Loading NLP models...")
intent_clf = TouristIntentClassifier.load()
retriever = DestinationRetriever()
translator = MultilingualTranslator()
recommender = TouristRecommender()
print("All NLP models loaded successfully!")


@app.route("/")
def index():
    return render_template("index.html", languages=SUPPORTED_LANGUAGES)


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    query = data.get("query", "").strip()
    user_lang = data.get("user_lang", "en") # e.g. 'en', 'es', 'fr', 'de', 'ja', 'hi'

    if not query:
        return jsonify({"error": "Empty query"}), 400

    # If query is not in English, translate query to English for NLP retrieval
    query_in_en = query
    if user_lang != "en":
        trans_res = translator.translate(query, src_lang=user_lang, tgt_lang="en")
        query_in_en = trans_res.get("translated_text", query)

    # 1. NLP Intent Classification & Entity Extraction
    intent_res = intent_clf.predict(query_in_en)

    # 2. Semantic Document Retrieval & Answer Synthesis
    qa_res = retriever.answer_query(query_in_en, intent_res)
    answer_en = qa_res["answer"]

    # 3. Translate answer back to user's native language if needed
    final_answer = answer_en
    if user_lang != "en":
        trans_back = translator.translate(answer_en, src_lang="en", tgt_lang=user_lang)
        final_answer = trans_back.get("translated_text", answer_en)

    return jsonify({
        "original_query": query,
        "processed_query_en": query_in_en,
        "intent": intent_res["intent"],
        "confidence": intent_res["confidence"],
        "entities": intent_res["entities"],
        "answer": final_answer,
        "matched_destination": qa_res.get("matched_destination"),
        "city": qa_res.get("city")
    })


@app.route("/api/translate", methods=["POST"])
def translate_api():
    data = request.get_json() or {}
    text = data.get("text", "").strip()
    src_lang = data.get("src_lang", "en")
    tgt_lang = data.get("tgt_lang", "hi")
    force_offline = bool(data.get("offline", False))

    res = translator.translate(text, src_lang=src_lang, tgt_lang=tgt_lang, force_offline=force_offline)
    return jsonify(res)


@app.route("/api/recommend", methods=["POST"])
def recommend_api():
    data = request.get_json() or {}
    city = data.get("city", "Delhi")
    category = data.get("category", "All")
    user_lat = data.get("lat")
    user_lon = data.get("lon")

    recs = recommender.recommend(
        user_lat=user_lat,
        user_lon=user_lon,
        city=city,
        category=category,
        top_n=6
    )
    return jsonify({"recommendations": recs, "city": city, "category": category})


@app.route("/api/phrases", methods=["GET"])
def phrases_api():
    category = request.args.get("category", "All")
    phrases = translator.get_phrasebook_by_category(category)
    return jsonify({
        "categories": translator.categories,
        "phrases": phrases
    })


@app.route("/api/metrics", methods=["GET"])
def metrics_api():
    metrics_path = os.path.join(os.path.dirname(__file__), "data", "evaluation_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
        return jsonify(metrics_data)
    return jsonify({"error": "Metrics not found. Run evaluate.py first."}), 404


if __name__ == "__main__":
    # Run development server on port 5000
    app.run(host="0.0.0.0", port=5000, debug=False)
