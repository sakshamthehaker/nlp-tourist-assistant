# TourLingo AI: Multilingual Assistant for Tourist Information & Local Language Translation
**Project ID:** `CSEAIML2026-P26`  
**Course Domain:** Natural Language Processing (NLP) / AI & Machine Learning  

---

## 📌 Executive Summary
**TourLingo AI** is an intelligent travel-assistant designed to bridge linguistic and cultural barriers for domestic and international travelers. Built following rigorous NLP principles, the application combines **Statistical Intent Classification**, **Vector Space Information Retrieval (TF-IDF Cosine Similarity)**, **Two-Way Neural Machine Translation (NMT)**, **Haversine Geodesic Recommendations**, and an **Offline-Friendly Jaccard/Levenshtein Survival Phrasebook**.

---

## 🏛️ System Architecture

```
                       ┌──────────────────────────────────────────────┐
                       │     Frontend UI (Tailwind + Web Speech)      │
                       │  (Tourist View | Vendor View | Audio TTS)    │
                       └──────────────────────┬───────────────────────┘
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
         ┌─────────────────────┐   ┌─────────────────────┐   ┌─────────────────────┐
         │ 1. Multilingual Q&A │   │ 2. Dual Translator  │   │ 3. Geo Recommender  │
         │ ─────────────────── │   │ ─────────────────── │   │ ─────────────────── │
         │ • TF-IDF Vectorizer │   │ • NMT (Google/NLLB) │   │ • Haversine formula │
         │ • Intent Classifier │   │ • Two-Way dialogue  │   │ • Rating & utility  │
         │ • Cosine Retrieval  │   │ • Offline Fallback  │   │ • Proximity ranking │
         └─────────────────────┘   └─────────────────────┘   └─────────────────────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              ▼
                               ┌─────────────────────────────┐
                               │ 4. Offline Phrasebook & SOS │
                               │ • Levenshtein/Jaccard match │
                               │ • Roman Phonetics & Audio   │
                               │ • Emergency Helplines (112) │
                               └─────────────────────────────┘
                                              │
                                              ▼
                               ┌─────────────────────────────┐
                               │ 5. NLP Evaluation Suite     │
                               │ • SacreBLEU & chrF metrics  │
                               │ • Intent Confusion Matrix   │
                               │ • Recommendation MRR & P@K  │
                               └─────────────────────────────┘
```

---

## 🧠 NLP Core Concepts Implemented

1. **Text Preprocessing & Normalization (`src/preprocessing.py`)**:
   - Tokenization via regex boundary splitting.
   - Text casing normalization, contraction handling, and punctuation removal.
   - Stopword filtering using custom domain-preserving lists (retaining question words like *how*, *where*, *when* crucial for query semantics).
   - Stemming using NLTK's `PorterStemmer`.

2. **Intent Classification & Entity Extraction (`src/intent_classifier.py`)**:
   - Supervised classification across 8 core tourist intent classes (`ask_attraction`, `ask_transport`, `ask_etiquette`, `food_recommendation`, `emergency_sos`, `currency_budget`, `bargaining_vendor`, `general_greeting`).
   - Unigram and bigram TF-IDF representations with sublinear TF scaling.
   - Multi-class Logistic Regression with balanced class weighting.
   - Slot-filling rule extractor identifying target city (Delhi, Agra, Jaipur, Varanasi), landmark (POI), and topic.

3. **Information Retrieval & Semantic Q&A (`src/semantic_retriever.py`)**:
   - Vector space document matching using TF-IDF and Cosine Similarity over structured destination knowledge.
   - Synthesizes formatted answers incorporating transport routes, opening hours, cultural etiquette, and emergency contacts.

4. **Neural Machine Translation & Offline Fallback (`src/translator.py`)**:
   - Bidirectional online NMT between English, Hindi, Spanish, French, German, and Japanese.
   - Zero-network offline fallback using token Jaccard similarity and SequenceMatcher string distance over cached survival phrases.
   - Generates Roman phonetic transliteration so tourists can pronounce Hindi phrases to local vendors.

5. **Location-Based Recommender (`src/recommender.py`)**:
   - Computes great-circle distances in kilometers via the Haversine formula:
     $$\text{distance} = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
   - Combines geographic proximity and quality ratings into a unified relevance score.

---

## 📊 Academic Evaluation Metrics

The system was evaluated using standard NLP benchmarks via `src/evaluate.py`:

| Evaluation Category | Metric | Achieved Result | Notes |
| :--- | :--- | :--- | :--- |
| **Intent Classification** | Test Accuracy | **53.85%** | 8 diverse classes on stratified test split |
| **Intent Classification** | General Greeting F1 | **1.00** | Perfect precision & recall |
| **Intent Classification** | Food Recommendation F1| **0.67** | High relevance for dining queries |
| **Translation Quality** | BLEU (EN ➔ HI) | **22.68** | SacreBLEU standard corpus score |
| **Translation Quality** | chrF (EN ➔ HI) | **48.10** | Character n-gram F-score |
| **Translation Quality** | BLEU (EN ➔ FR) | **28.27** | High translation fluency |
| **Recommendation Engine**| Hit Rate@3 | **100.0%** | Relevant destination in top 3 results |
| **Recommendation Engine**| Precision@3 | **73.33%** | Mean top-3 accuracy across cities |
| **Recommendation Engine**| MRR | **1.0000** | Gold standard rank at position 1 |

*Visual charts saved at `static/evaluation_charts.png`.*

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Evaluation Suite (Produces Metrics & Plots)
```bash
python src/evaluate.py
```

### 3. Run the Demonstration of Representative Tourist Scenarios
```bash
python demo_scenarios.py
```

### 4. Launch the Interactive Web Application
```bash
python app.py
```
Open your browser and navigate to: **`http://localhost:5000`**

---

## 📂 Project Directory Structure

```text
nlp/
├── data/
│   ├── destination_knowledge.json    # Curated knowledge base (monuments, transport, customs)
│   ├── tourist_intents.json          # Labeled training dataset for Intent Classification
│   ├── offline_phrases.json          # Offline multilingual survival phrasebook
│   ├── intent_model.joblib           # Serialized trained NLP intent pipeline
│   └── evaluation_metrics.json       # Exported quantitative evaluation results
├── src/
│   ├── preprocessing.py              # Cleaning, tokenization, stopword filter & stemming
│   ├── intent_classifier.py          # TF-IDF + Logistic Regression Classifier & NER
│   ├── semantic_retriever.py         # TF-IDF Cosine Similarity Q&A engine
│   ├── translator.py                 # NMT + Offline Levenshtein/Jaccard matcher
│   ├── recommender.py                # Haversine distance & preference ranking
│   └── evaluate.py                   # SacreBLEU, chrF, Confusion Matrix generator
├── static/
│   └── evaluation_charts.png         # Generated publication-quality evaluation plot
├── templates/
│   └── index.html                    # Modern responsive web interface (Tailwind + Web Speech)
├── app.py                            # Flask server exposing REST APIs and Web UI
├── demo_scenarios.py                 # Representative tourist scenarios test harness
├── test_app.py                       # Automated end-to-end unit tests
├── requirements.txt                  # Python dependencies
└── README.md                         # Technical documentation
```
