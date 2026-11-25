"""
evaluate.py
Formal Academic NLP Evaluation Suite.
Computes:
1. Machine Translation Quality: Corpus & Sentence BLEU, chrF using sacrebleu.
2. Intent Classification: Precision, Recall, F1-Score, Accuracy, and Confusion Matrix.
3. Recommendation Relevance: Precision@K, HitRate@K, Mean Reciprocal Rank (MRR).
Generates report and plots saved to static/ for viva and report presentation.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg') # Headless backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sacrebleu.metrics import BLEU, CHRF

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.intent_classifier import TouristIntentClassifier, load_intent_data
from src.translator import MultilingualTranslator
from src.recommender import TouristRecommender

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "static")
METRICS_JSON_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "evaluation_metrics.json")


def evaluate_intent_classifier():
    """Evaluates the Intent Classifier across Stratified train-test split."""
    print("\n================ 1. INTENT CLASSIFICATION EVALUATION ================")
    texts, labels = load_intent_data()
    
    # Stratified 80/20 split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.25, random_state=42, stratify=labels
    )
    
    clf = TouristIntentClassifier(model_type="logistic_regression")
    clf.train(X_train, y_train)
    
    y_pred = [clf.predict(x)["intent"] for x in X_test]
    
    acc = accuracy_score(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    classes = sorted(list(set(labels)))
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    
    print(f"Intent Classification Test Accuracy: {acc * 100:.2f}%")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    return {
        "accuracy": round(float(acc), 4),
        "classification_report": report_dict,
        "classes": classes,
        "confusion_matrix": cm.tolist()
    }


def evaluate_translation_bleu():
    """Evaluates Translation Quality using BLEU and chrF against gold references."""
    print("\n================ 2. MACHINE TRANSLATION EVALUATION ================")
    
    # Gold standard tourist benchmark test set
    test_benchmarks = [
        {
            "src": "How much does this cost?",
            "ref_hi": "यह कितने का है?",
            "ref_es": "¿Cuánto cuesta esto?",
            "ref_fr": "Combien cela coûte-t-il ?"
        },
        {
            "src": "Please give me a discount.",
            "ref_hi": "कृपया मुझे छूट दीजिए।",
            "ref_es": "Por favor deme un descuento.",
            "ref_fr": "S'il vous plaît donnez-moi une réduction."
        },
        {
            "src": "Where is the nearest railway station?",
            "ref_hi": "नज़दीकी रेलवे स्टेशन कहाँ है?",
            "ref_es": "¿Dónde está la estación de tren más cercana?",
            "ref_fr": "Où est la gare ferroviaire la plus proche ?"
        },
        {
            "src": "Please stop the car here.",
            "ref_hi": "कृपया गाड़ी यहाँ रोकिए।",
            "ref_es": "Por favor pare el coche aquí.",
            "ref_fr": "Arrêtez la voiture ici s'il vous plaît."
        },
        {
            "src": "Is this food vegetarian?",
            "ref_hi": "क्या यह खाना शाकाहारी है?",
            "ref_es": "¿Esta comida es vegetariana?",
            "ref_fr": "Cette nourriture est-elle végétarienne ?"
        },
        {
            "src": "I need help immediately.",
            "ref_hi": "मुझे तुरंत मदद की ज़रूरत है।",
            "ref_es": "Necesito ayuda de inmediato.",
            "ref_fr": "J'ai besoin d'aide immédiatement."
        }
    ]
    
    translator = MultilingualTranslator()
    bleu_metric = BLEU(effective_order=True)
    chrf_metric = CHRF()
    
    results = {}
    for tgt_lang, ref_key in [("hi", "ref_hi"), ("es", "ref_es"), ("fr", "ref_fr")]:
        hypotheses = []
        references = []
        for item in test_benchmarks:
            # Use offline phrasebook or online translation
            trans_res = translator.translate(item["src"], src_lang="en", tgt_lang=tgt_lang)
            hypotheses.append(trans_res["translated_text"])
            references.append(item[ref_key])
        
        bleu_score = bleu_metric.corpus_score(hypotheses, [references]).score
        chrf_score = chrf_metric.corpus_score(hypotheses, [references]).score
        
        results[tgt_lang] = {
            "bleu": round(float(bleu_score), 2),
            "chrf": round(float(chrf_score), 2),
            "samples": [
                {"src": s["src"], "hyp": h, "ref": r}
                for s, h, r in zip(test_benchmarks, hypotheses, references)
            ]
        }
        print(f"Language Pair en -> {tgt_lang.upper()}: BLEU = {bleu_score:.2f} | chrF = {chrf_score:.2f}")
    
    return results


def evaluate_recommendation_engine():
    """Evaluates Recommendation Relevance (Precision@K, Hit Rate@K, MRR)."""
    print("\n================ 3. RECOMMENDATION ENGINE EVALUATION ================")
    recommender = TouristRecommender()
    
    # Ground truth test cases: (query_city, query_category, expected_relevant_substrings)
    test_cases = [
        ("Delhi", "Attraction", ["Qutub Minar", "Red Fort", "Humayun's Tomb"]),
        ("Agra", "Attraction", ["Taj Mahal", "Agra Fort"]),
        ("Jaipur", "Attraction", ["Amber Palace", "Hawa Mahal", "City Palace"]),
        ("Delhi", "Food & Dining", ["Paranthe Wali Gali", "Karim's"]),
        ("Varanasi", "Cultural / Spiritual", ["Dashashwamedh Ghat"])
    ]
    
    k = 3
    precision_scores = []
    hit_scores = []
    reciprocal_ranks = []
    
    for city, cat, ground_truth in test_cases:
        recs = recommender.recommend(city=city, category=cat, top_n=k)
        retrieved_names = [r["name"] for r in recs]
        
        # Check hits
        hits = 0
        first_hit_rank = 0
        for rank, name in enumerate(retrieved_names, start=1):
            if any(gt.lower() in name.lower() for gt in ground_truth):
                hits += 1
                if first_hit_rank == 0:
                    first_hit_rank = rank
                    
        prec = hits / k if k > 0 else 0
        precision_scores.append(prec)
        hit_scores.append(1 if hits > 0 else 0)
        reciprocal_ranks.append(1.0 / first_hit_rank if first_hit_rank > 0 else 0.0)
    
    mean_p = round(float(np.mean(precision_scores)), 4)
    hit_rate = round(float(np.mean(hit_scores)), 4)
    mrr = round(float(np.mean(reciprocal_ranks)), 4)
    
    print(f"Mean Precision@{k}: {mean_p * 100:.2f}%")
    print(f"Hit Rate@{k}: {hit_rate * 100:.2f}%")
    print(f"Mean Reciprocal Rank (MRR): {mrr:.4f}")
    
    return {
        "k": k,
        "precision_at_k": mean_p,
        "hit_rate_at_k": hit_rate,
        "mrr": mrr
    }


def generate_evaluation_plots(intent_res, trans_res, rec_res):
    """Plots and saves academic evaluation charts."""
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Confusion Matrix
    cm = np.array(intent_res["confusion_matrix"])
    labels = [c.replace("ask_", "").replace("_", " ").title() for c in intent_res["classes"]]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Intent Classification Confusion Matrix", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Predicted Label", fontsize=11)
    axes[0].set_ylabel("True Label", fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    
    # 2. BLEU & chrF Translation Metrics
    lang_pairs = [f"EN->{k.upper()}" for k in trans_res.keys()]
    bleu_vals = [trans_res[k]["bleu"] for k in trans_res.keys()]
    chrf_vals = [trans_res[k]["chrf"] for k in trans_res.keys()]
    
    x = np.arange(len(lang_pairs))
    width = 0.35
    
    axes[1].bar(x - width/2, bleu_vals, width, label='BLEU Score', color='#2563EB')
    axes[1].bar(x + width/2, chrf_vals, width, label='chrF Score', color='#10B981')
    axes[1].set_ylabel('Metric Score (0-100)', fontsize=11)
    axes[1].set_title('Neural Translation Quality Across Target Languages', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(lang_pairs)
    axes[1].set_ylim(0, 100)
    axes[1].legend(loc='upper right')
    axes[1].grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotate bars
    for i in range(len(lang_pairs)):
        axes[1].text(x[i] - width/2, bleu_vals[i] + 1.5, f"{bleu_vals[i]}", ha='center', fontsize=9, fontweight='bold')
        axes[1].text(x[i] + width/2, chrf_vals[i] + 1.5, f"{chrf_vals[i]}", ha='center', fontsize=9, fontweight='bold')
        
    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, "evaluation_charts.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"\n📊 Evaluation charts generated and saved to: {chart_path}")


def run_full_evaluation():
    intent_eval = evaluate_intent_classifier()
    trans_eval = evaluate_translation_bleu()
    rec_eval = evaluate_recommendation_engine()
    
    full_report = {
        "project_id": "CSEAIML2026-P26",
        "project_title": "Multilingual AI Assistant for Tourist Information and Local Language Translation",
        "intent_classification": intent_eval,
        "machine_translation": trans_eval,
        "recommendation_engine": rec_eval
    }
    
    with open(METRICS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"Metrics report exported to: {METRICS_JSON_PATH}")
    
    generate_evaluation_plots(intent_eval, trans_eval, rec_eval)
    return full_report


if __name__ == "__main__":
    run_full_evaluation()
