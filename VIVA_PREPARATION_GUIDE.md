# TourLingo AI: Project Viva & Presentation Preparation Guide
**Project ID:** `CSEAIML2026-P26`  
**Title:** Multilingual AI Assistant for Tourist Information & Local Language Translation  
**Domain:** Natural Language Processing (NLP) & AI Engineering  

---

## 📌 Team Role Division Overview

| Member | Focus Domain | Files Owned | Primary Metric |
| :--- | :--- | :--- | :--- |
| **Member 1** | Intent Detection & NLP Preprocessing | `src/preprocessing.py`<br>`src/intent_classifier.py`<br>`data/tourist_intents.json` | Intent Accuracy: **72.22%**<br>Macro Precision: **78%** |
| **Member 2** | Neural Machine Translation & Offline Matching | `train_nmt.py`<br>`src/translator.py`<br>`models/custom_en_hi_model/`<br>`data/offline_phrases.json` | 1,875 Training Steps<br>Loss Drop: **3.37 ➔ 2.78**<br>Peak Val BLEU: **9.17** |
| **Member 3** | Semantic Retrieval (RAG), Recommender & Full-Stack UI | `src/semantic_retriever.py`<br>`src/recommender.py`<br>`app.py` & `templates/index.html`<br>`src/evaluate.py` & `test_app.py` | Recommendation Hit@3: **100%**<br>MRR: **1.0000**<br>Precision@3: **73.33%** |

---

# 👤 MEMBER 1: NLP Preprocessing, Intent Classifier & Slot-Filling

### 📁 Files to Review:
* `src/preprocessing.py`
* `src/intent_classifier.py`
* `data/tourist_intents.json`

### 🎯 Key Contributions to State in Viva:
1. **Text Preprocessing:** Built clean pipeline using regex tokenization, stopword whitelisting (retained crucial interrogatives like *how*, *where*, *when*), and NLTK's `PorterStemmer`.
2. **Feature Extraction:** Implemented sublinear TF-IDF scaling ($1 + \log(\text{TF})$) across Unigrams, Bigrams, and Trigrams $(1, 3)$.
3. **Intent Classifier:** Trained a **Calibrated Linear Support Vector Machine (`CalibratedClassifierCV(LinearSVC)`)** over 214 tourist query samples across 8 intent classes.
4. **Entity Extraction (NER):** Implemented slot-filling to extract mentioned cities (Delhi, Agra, Jaipur, Varanasi), monument POIs (Taj Mahal, Qutub Minar), and travel topics.

### ❓ Top Viva Questions & Answers:
* **Q1: Why use Trigram TF-IDF $(1, 3)$ instead of just Unigrams?**  
  * **Answer:** Unigrams only see single words (e.g. *"ticket"*). Trigrams capture full multi-word travel phrases (e.g. *"light and sound"*, *"free entry fee"*, *"how to reach"*), preventing confusion between overlapping travel questions.
* **Q2: Why choose LinearSVC over standard Logistic Regression?**  
  * **Answer:** TF-IDF feature spaces are high-dimensional and sparse. Linear SVMs maximize the geometric margin between classes, preventing overfitting and giving higher generalization accuracy on text.
* **Q3: Why is `CalibratedClassifierCV` needed with LinearSVC?**  
  * **Answer:** Standard `LinearSVC` does not output probabilities, only raw geometric distances. CalibratedClassifierCV fits Platt scaling to give true calibrated probability percentages for user confidence scores.

---

# 👤 MEMBER 2: Neural Machine Translation (NMT), Deep Learning & Offline Algorithms

### 📁 Files to Review:
* `train_nmt.py` & `train_nmt_colab.ipynb`
* `src/translator.py`
* `models/custom_en_hi_model/` (303 MB trained model)
* `data/offline_phrases.json`

### 🎯 Key Contributions to State in Viva:
1. **Deep Learning Seq2Seq Model:** Fine-tuned MarianMT (`opus-mt-en-hi`) Transformer model (6 encoder layers + 6 decoder layers, multi-head self-attention).
2. **Training Pipeline:** Ingested 10,000 parallel sentence pairs from the **IIT Bombay English-Hindi Corpus (`cfilt/iitb-english-hindi`)**. Trained across 3 epochs (1,875 global steps) using PyTorch, AdamW optimizer ($\text{lr}=2\times 10^{-5}$), and Hugging Face `Seq2SeqTrainer`.
3. **Training Results:** Loss dropped from **3.37 down to 2.78** (17.6% reduction) with a peak validation BLEU of **9.17** at step 1,250.
4. **Offline Zero-Network Fallback:** Built an offline phrase matcher combining Ratcliff-Obershelp string ratio (`difflib.SequenceMatcher`, 60% weight) and Token Jaccard Similarity (40% weight) over categorized survival phrases, providing Roman phonetic transliteration (*"Yeh kitne ka hai?"*).

### ❓ Top Viva Questions & Answers:
* **Q1: What is the underlying architecture of MarianMT?**  
  * **Answer:** It is a Sequence-to-Sequence Transformer with an Encoder-Decoder structure. Both encoder and decoder have 6 layers with multi-head self-attention, tokenized using SentencePiece Byte-Pair Encoding (BPE).
* **Q2: What is the BLEU score and how is it calculated?**  
  * **Answer:** BLEU (Bilingual Evaluation Understudy) measures n-gram precision overlap between machine output and gold human references, scaled by a brevity penalty. We used the standardized `sacrebleu` library.
* **Q3: How does the offline mode work without internet?**  
  * **Answer:** When internet is disconnected, queries are routed to our offline phrasebook matcher that computes fuzzy character distance and token Jaccard overlap to find the closest survival phrase and its phonetic pronunciation.

---

# 👤 MEMBER 3: Semantic Q&A Retrieval (RAG), Geo-Spatial Recommender & Full-Stack UI

### 📁 Files to Review:
* `src/semantic_retriever.py`
* `src/recommender.py`
* `app.py` & `templates/index.html`
* `src/evaluate.py` & `test_app.py`

### 🎯 Key Contributions to State in Viva:
1. **Granular Passage Chunking & Semantic Q&A:** Split destination records into discrete semantic chunks (`overview`, `timing_pricing`, `transport`, `etiquette`, `emergency`) and applied Cosine Similarity in TF-IDF Vector Space for exact facet matching.
2. **Geo-Spatial Location Recommender:** Implemented the spherical Haversine formula to compute great-circle kilometer distances from GPS coordinates, ranked via multi-attribute utility scoring (combining distance + review ratings).
3. **Full-Stack Application & Speech Integration:** Built the Flask REST API server and modern Tailwind UI with in-browser Web Speech API for voice input (Speech-to-Text) and audio playback (Text-to-Speech).
4. **Benchmarking & Unit Testing:** Implemented automated evaluation scripts generating `evaluation_charts.png` (Confusion Matrix & BLEU charts) and `test_app.py` with 6 passing unit tests.

### ❓ Top Viva Questions & Answers:
* **Q1: Why use granular passage chunking instead of full-page retrieval?**  
  * **Answer:** Indexing an entire monument page dilutes the vector. By breaking it into semantic chunks (`[Taj Mahal | Timings]`, `[Taj Mahal | Transport]`), query vectors achieve higher Cosine Similarity with the exact relevant section.
* **Q2: What is the Haversine formula and why is it needed?**  
  * **Answer:** The Earth is spherical, so flat Euclidean distance causes significant distortion. The Haversine formula computes the true great-circle distance over the Earth's curvature using spherical trigonometry with radius $R = 6,371\text{ km}$.
* **Q3: What is Mean Reciprocal Rank (MRR) and what did you achieve?**  
  * **Answer:** MRR evaluates where the first relevant item appears in a ranked list ($\text{MRR} = \frac{1}{|Q|}\sum \frac{1}{\text{rank}_i}$). Our recommender achieved an **MRR of 1.0000** and **100% Hit Rate@3**, placing the gold standard match at rank 1.

---

## 🚀 Quick Commands to Demonstrate in Viva

```bash
# 1. Run the 6 representative tourist scenarios demo:
python demo_scenarios.py

# 2. Run automated unit tests (All 6 pass):
python test_app.py

# 3. Launch interactive Web App:
python app.py
# Open browser at: http://localhost:5000
```
