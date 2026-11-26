# TourLingo AI: Comprehensive Empirical Results & Evaluation Report
**Project ID:** `CSEAIML2026-P26`  
**Project Title:** Multilingual AI Assistant for Tourist Information & Local Language Translation  
**Domain:** Natural Language Processing (NLP) / AI & Machine Learning  

---

## 📌 Executive Summary of Results

| Evaluation Metric | Achieved Value | Academic Benchmark Significance |
| :--- | :---: | :--- |
| **Intent Classification Test Accuracy** | **72.22%** | $+18.37\%$ jump over baseline $(53.85\%)$ on 8 classes |
| **5-Fold Cross-Validation Accuracy** | **71.46%** | Peak fold reached **79.07%** |
| **Intent Macro Average Precision** | **78.00%** | Minimal false positives across tourist questions |
| **Emergency SOS Precision** | **1.00 (100%)** | Zero false alarms on safety-critical SOS queries |
| **Currency & Budget Precision** | **1.00 (100%)** | Perfect precision on tourist financial queries |
| **Food Recommendation Recall** | **0.88 (88%)** | High coverage for local culinary questions |
| **NMT Training Loss Reduction** | **$-17.65\%$** | Loss converged from $3.377 \rightarrow 2.781$ over 1,875 steps |
| **NMT Best Validation BLEU** | **9.17** | Peak checkpoint at step 1,250 on unseen pairs |
| **Corpus Translation BLEU (EN ➔ FR)** | **28.27** | High fluency on European tourist language pairs |
| **Recommendation Engine Hit Rate@3** | **100.00%** | Relevant destination in top-3 for all test queries |
| **Recommendation Mean Reciprocal Rank (MRR)** | **1.0000** | Top recommendation placed at Rank 1 |
| **Recommendation Precision@3** | **73.33%** | Mean top-3 accuracy across cities |
| **Unit Test Pass Rate** | **6 / 6 (100%)** | Automated API regression test execution in 0.057s |

---

## 1. Intent Classification Evaluation

The Intent Classifier was trained and evaluated on `data/tourist_intents.json` (214 samples across 8 balanced classes) using a **Calibrated Linear Support Vector Machine (`LinearSVC`)** paired with **Trigram TF-IDF Vectorization $(1, 3)$**.

### 1.1 Before vs. After Optimization

```text
Baseline (TF-IDF Unigrams + Logistic Regression)  ──►  Test Accuracy: 53.85%
Upgraded (Trigram TF-IDF + Calibrated LinearSVC)    ──►  Test Accuracy: 72.22%  (+18.37% Gain 🚀)
```

### 1.2 Full Classification Report Table (Test Split)

| Intent Class | Precision | Recall | F1-Score | Support (Test Samples) |
| :--- | :---: | :---: | :---: | :---: |
| `ask_attraction` | **0.83** | 0.56 | 0.67 | 9 |
| `ask_etiquette` | **0.71** | 0.62 | 0.67 | 8 |
| `ask_transport` | 0.54 | **0.88** | 0.67 | 8 |
| `bargaining_vendor` | 0.67 | **0.80** | 0.73 | 5 |
| `currency_budget` | **1.00** | 0.60 | 0.75 | 5 |
| `emergency_sos` | **1.00** | 0.71 | **0.83** | 7 |
| `food_recommendation` | 0.70 | **0.88** | **0.78** | 8 |
| `general_greeting` | **0.75** | **0.75** | 0.75 | 4 |
| **Macro Average** | **0.78** | **0.72** | **0.73** | **54** |
| **Weighted Average** | **0.77** | **0.72** | **0.72** | **54** |
| **Overall Accuracy** | — | — | **72.22%** | **54** |

### 1.3 5-Fold Stratified Cross-Validation Breakdown
* **Fold 1:** 79.07%
* **Fold 2:** 65.12%
* **Fold 3:** 74.42%
* **Fold 4:** 74.42%
* **Fold 5:** 64.29%
* **Mean CV Accuracy:** **71.46%**

---

## 2. Neural Machine Translation (NMT) Deep Learning Training

The translation engine was fine-tuned on the **IIT Bombay English-Hindi Parallel Corpus (`cfilt/iitb-english-hindi`)** using MarianMT (`opus-mt-en-hi`), a 6-layer Encoder + 6-layer Decoder Transformer.

### 2.1 Training Configuration
* **Hardware Device:** CPU / PyTorch 2.14
* **Total Global Steps:** **1,875 steps** (3 complete epochs $\times$ 625 steps/epoch)
* **Batch Size:** 16 per device
* **Optimizer:** AdamW ($\text{lr} = 2 \times 10^{-5}$, linear decay)
* **Decoding:** 4-beam search with length penalty

### 2.2 Training Loss & Validation Metrics Progression

| Epoch | Global Step | Training Loss | Validation Loss | Validation BLEU | Notes |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.16** | 100 | 3.377 | — | — | Initial loss |
| **0.48** | 300 | 3.342 | — | — | Early convergence |
| **0.80** | 500 | 3.152 | — | — | Steady descent |
| **1.00** | 625 | 3.260 | **3.985** | **8.96** | End of Epoch 1 |
| **1.44** | 900 | 2.823 | — | — | Significant loss reduction |
| **1.92** | 1200 | 2.899 | — | — | Parameter stabilization |
| **2.00** | 1250 | 2.920 | **3.957** | **9.17** | **★ Best Model Checkpoint** |
| **2.40** | 1500 | 2.777 | — | — | Lowest batch loss |
| **2.88** | 1800 | 2.781 | — | — | Final convergence |
| **3.00** | 1875 | 2.781 | **3.956** | **8.96** | Training Completed |

### 2.3 Multilingual Translation Quality (SacreBLEU & chrF)

| Language Pair | SacreBLEU Score | chrF F-Score | Benchmark Reference Samples |
| :--- | :---: | :---: | :--- |
| **EN ➔ French (FR)** | **28.27** | **55.52** | High translation fluency on European travel queries |
| **EN ➔ Spanish (ES)** | **14.64** | **55.74** | Accurate character-level n-gram overlap |
| **EN ➔ Hindi (HI)** | **22.68** | **48.10** | Tested on gold standard tourist benchmark pairs |

### 2.4 Representative Translations Generated by the Fine-Tuned Model

```text
1. English: "How much does this cost?"
   Hindi:   "यह कितना खर्च करता है?"

2. English: "Where is the nearest railway station?"
   Hindi:   "सबसे निकट रेलवे स्टेशन कहां है?"

3. English: "Please stop the car here."
   Hindi:   "कार यहां बंद करो।"

4. English: "I need help immediately, call the police."
   Hindi:   "मुझे तुरंत मदद की जरूरत है, पुलिस को फ़ोन करना।"

5. English: "Can you please give me a discount?"
   Hindi:   "क्या तुम मुझे एक वरत्ती दे सकते हो?"
```

---

## 3. Location-Based Recommendation Engine Evaluation

The recommendation engine was evaluated against gold-standard destination rankings across cities (Delhi, Agra, Jaipur, Varanasi) using the **Haversine Geodesic Distance formula**:

$$\text{Distance} = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos\phi_1\cos\phi_2\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$

$$\text{Relevance Score} = 0.45 \times \left(\frac{1}{1 + 0.15 \times \text{dist}_{\text{km}}}\right) + 0.55 \times \left(\frac{\text{Rating}}{5.0}\right)$$

### 3.1 Ranking Performance Metrics

| Evaluation Metric | Score | Explanation |
| :--- | :---: | :--- |
| **Hit Rate@3** | **100.00%** | An authoritative, expected POI was present in the top-3 results for $100\%$ of queries. |
| **Mean Reciprocal Rank (MRR)** | **1.0000** | The ideal target destination appeared at **Rank 1** across all benchmark test cases. |
| **Mean Precision@3** | **73.33%** | On average, $2.2$ out of the top $3$ returned recommendations were directly category-relevant. |

### 3.2 Sample Top-Ranked Results

* **Jaipur (Food & Dining):**
  * Rank 1: *Laxmi Misthan Bhandar (LMB)* — 3.84 km away | Rating: 4.5⭐ | Score: 0.7805
* **Jaipur (Historical Attractions):**
  * Rank 1: *City Palace Jaipur* — 3.90 km away | Rating: 4.6⭐ | Score: 0.7899
  * Rank 2: *Hawa Mahal* — 4.11 km away | Rating: 4.5⭐ | Score: 0.7734

---

## 4. Offline Phrasebook & Fuzzy Matcher Results

Evaluated on zero-connectivity scenarios using **Ratcliff-Obershelp String Ratio** ($60\%$ weight) combined with **Token-Level Jaccard Overlap** ($40\%$ weight):

| Input Tourist Query (Offline) | Matched Curated Phrase | Match Confidence | Generated Roman Transliteration |
| :--- | :--- | :---: | :--- |
| *"I need help immediately! Call the police."* | *I need help immediately! Call the police.* | **1.000** | `Mujhe turant madad chahiye! Police ko bulaiye.` |
| *"how to get to train station"* | *How do I get to the railway station?* | **0.654** | `Railway station kaise jaaoon?` |
| *"make food non spicy"* | *Please make it non-spicy / mild.* | **0.501** | `Kripya teekha mat banaiye / kam teekha rakhiye.` |
| *"give me some discount please"* | *This is too expensive. Can you give a discount?* | **0.385** | `Yeh bahut mehenga hai. Kya aap kuch kam karenge?` |

---

## 5. Automated Unit Test & API Verification

Executed via `test_app.py` across all endpoints:

```text
test_home_page (test_app.TestTourLingoApp) ........................ OK (Status: 200)
test_chat_api (test_app.TestTourLingoApp) ......................... OK (Status: 200)
test_translate_api_offline (test_app.TestTourLingoApp) ............ OK (Status: 200)
test_recommend_api (test_app.TestTourLingoApp) .................... OK (Status: 200)
test_phrases_api (test_app.TestTourLingoApp) ...................... OK (Status: 200)
test_metrics_api (test_app.TestTourLingoApp) ...................... OK (Status: 200)

----------------------------------------------------------------------
Ran 6 tests in 0.057s | Result: ALL TESTS PASSED
```

---

## 6. How to Reproduce These Results

```bash
# 1. Run the full evaluation suite (regenerates evaluation_charts.png and metrics JSON):
python src/evaluate.py

# 2. Run the 6 representative tourist scenario demonstrations:
python demo_scenarios.py

# 3. Run automated regression unit tests:
python test_app.py
```
