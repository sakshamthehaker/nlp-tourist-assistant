"""
intent_classifier.py
High-Accuracy Intent Classifier for Tourist Assistant using
N-gram (1-3) TF-IDF + Calibrated LinearSVC (Support Vector Machine).
Includes slot-filling entity extraction.
"""

import os
import sys
import json
import joblib
import warnings
import numpy as np

# Suppress sklearn unpickling warnings across different patch versions
warnings.filterwarnings("ignore", category=UserWarning)

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from src.preprocessing import preprocess

KNOWN_CITIES = ["delhi", "agra", "jaipur", "varanasi", "mumbai", "goa"]
INTENT_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tourist_intents.json")
MODEL_SAVE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "intent_model.joblib")


def load_intent_data(filepath=INTENT_DATA_PATH):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    texts = []
    labels = []
    for item in data["intents"]:
        intent = item["intent"]
        for example in item["examples"]:
            texts.append(example)
            labels.append(intent)
    return texts, labels


class TouristIntentClassifier:
    def __init__(self, model_type="svm"):
        self.model_type = model_type
        self.pipeline = None
        self.classes_ = []

    def build_pipeline(self):
        if self.model_type == "logistic_regression":
            base_clf = LogisticRegression(C=2.5, max_iter=1000, class_weight="balanced", random_state=42)
            clf = base_clf
        else: # Support Vector Machine with probability calibration
            base_svc = LinearSVC(C=1.2, class_weight="balanced", random_state=42)
            clf = CalibratedClassifierCV(estimator=base_svc, cv=3)

        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                preprocessor=preprocess,
                ngram_range=(1, 3), # Unigrams, bigrams, and trigrams
                sublinear_tf=True,
                min_df=1
            )),
            ("clf", clf)
        ])

    def train(self, texts, labels):
        self.build_pipeline()
        self.pipeline.fit(texts, labels)
        if hasattr(self.pipeline.named_steps["clf"], "classes_"):
            self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def cross_validate(self, texts, labels, n_splits=5):
        if self.pipeline is None:
            self.build_pipeline()
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(self.pipeline, texts, labels, cv=skf, scoring="accuracy")
        return float(np.mean(scores)), [float(s) for s in scores]

    def predict(self, query: str):
        if self.pipeline is None:
            raise ValueError("Model is not trained or loaded yet.")
        
        probas = self.pipeline.predict_proba([query])[0]
        max_idx = int(np.argmax(probas))
        intent = self.classes_[max_idx]
        confidence = float(probas[max_idx])
        
        all_scores = {cls_name: float(p) for cls_name, p in zip(self.classes_, probas)}
        
        # Slot / Entity extraction
        entities = self.extract_entities(query)

        return {
            "intent": intent,
            "confidence": round(confidence, 4),
            "all_scores": all_scores,
            "entities": entities
        }

    def extract_entities(self, query: str) -> dict:
        q_lower = query.lower()
        extracted = {
            "city": None,
            "topic": None,
            "poi": None
        }

        # City entity
        for c in KNOWN_CITIES:
            if c in q_lower:
                extracted["city"] = c.capitalize()
                break

        # Specific Landmark / POI detection
        pois = {
            "taj mahal": "Taj Mahal",
            "qutub minar": "Qutub Minar",
            "red fort": "Red Fort",
            "lal qila": "Red Fort",
            "humayun": "Humayun's Tomb",
            "bangla sahib": "Gurudwara Bangla Sahib",
            "amber fort": "Amber Palace",
            "amer fort": "Amber Palace",
            "hawa mahal": "Hawa Mahal",
            "city palace": "City Palace Jaipur",
            "ganga aarti": "Dashashwamedh Ghat",
            "dashashwamedh": "Dashashwamedh Ghat",
            "chandni chowk": "Paranthe Wali Gali"
        }
        for k, v in pois.items():
            if k in q_lower:
                extracted["poi"] = v
                break

        # Topic detection
        if any(w in q_lower for w in ["metro", "train", "bus", "taxi", "cab", "rickshaw", "auto", "reach", "fare"]):
            extracted["topic"] = "transport"
        elif any(w in q_lower for w in ["food", "restaurant", "eat", "thali", "dish", "cafe", "spicy", "dessert", "sweet"]):
            extracted["topic"] = "dining"
        elif any(w in q_lower for w in ["dress", "shoes", "temple", "custom", "etiquette", "tip", "scam", "safe", "greet"]):
            extracted["topic"] = "culture"
        elif any(w in q_lower for w in ["police", "lost", "hospital", "emergency", "doctor", "passport", "help", "sos", "ambulance"]):
            extracted["topic"] = "emergency"
        elif any(w in q_lower for w in ["cost", "ticket", "fee", "price", "upi", "card", "atm", "cash", "rupee", "budget"]):
            extracted["topic"] = "currency"

        return extracted

    def save(self, filepath=MODEL_SAVE_PATH):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "classes": self.classes_}, filepath)

    @classmethod
    def load(cls, filepath=MODEL_SAVE_PATH):
        try:
            if os.path.exists(filepath):
                data = joblib.load(filepath)
                clf = cls()
                clf.pipeline = data["pipeline"]
                clf.classes_ = data["classes"]
                clf.pipeline.predict_proba(["test"])
                return clf
        except Exception:
            pass

        # Self-healing training with current environment
        texts, labels = load_intent_data()
        clf = cls(model_type="svm")
        clf.train(texts, labels)
        try:
            clf.save(filepath)
        except Exception:
            pass
        return clf


def train_and_evaluate_classifier():
    texts, labels = load_intent_data()
    print(f"Total training dataset size: {len(texts)} samples across {len(set(labels))} intents.")
    
    classifier = TouristIntentClassifier(model_type="svm")
    mean_acc, scores = classifier.cross_validate(texts, labels, n_splits=5)
    print(f"5-Fold Stratified Cross-Validation Accuracy: {mean_acc * 100:.2f}% (folds: {[round(s, 3) for s in scores]})")

    # Fit full model & save
    classifier.train(texts, labels)
    classifier.save()
    print(f"Intent model successfully trained and saved to {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train_and_evaluate_classifier()
