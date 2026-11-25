"""
translator.py
Multilingual Real-Time Translation & Offline Survival Phrasebook Matcher.
Supports online NMT translation across multiple language pairs with automatic offline Levenshtein/Jaccard fallback.
"""

import os
import re
import json
from difflib import SequenceMatcher
from deep_translator import GoogleTranslator

OFFLINE_PHRASES_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "offline_phrases.json")
CUSTOM_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "custom_en_hi_model")

SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "es": "Spanish (Español)",
    "fr": "French (Français)",
    "de": "German (Deutsch)",
    "ja": "Japanese (日本語)"
}


def string_similarity(a: str, b: str) -> float:
    """Computes Ratcliff-Obershelp & Levenshtein-like similarity ratio (0.0 to 1.0)."""
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


def jaccard_similarity(a: str, b: str) -> float:
    """Computes word-level Jaccard token overlap similarity."""
    set_a = set(re.findall(r'\b\w+\b', a.lower()))
    set_b = set(re.findall(r'\b\w+\b', b.lower()))
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


class MultilingualTranslator:
    def __init__(self, offline_path=OFFLINE_PHRASES_PATH):
        self.offline_path = offline_path
        self.categories = []
        self.phrases = []
        self.load_offline_phrasebook()

    def load_offline_phrasebook(self):
        with open(self.offline_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.categories = data.get("categories", [])
        self.phrases = data.get("phrases", [])

    def translate_online(self, text: str, src_lang: str = "en", tgt_lang: str = "hi") -> dict:
        """
        Translates text via custom fine-tuned model (if available) or GoogleTranslator.
        """
        if not text or not text.strip():
            return {"translated_text": "", "mode": "online", "status": "empty"}

        if src_lang == tgt_lang:
            return {"translated_text": text, "mode": "online", "status": "identity"}

        # 1. Use locally fine-tuned Seq2Seq model if present
        if src_lang == "en" and tgt_lang == "hi" and os.path.exists(CUSTOM_MODEL_DIR):
            try:
                import torch
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
                if not hasattr(self, "_custom_model"):
                    self._custom_tokenizer = AutoTokenizer.from_pretrained(CUSTOM_MODEL_DIR)
                    self._custom_model = AutoModelForSeq2SeqLM.from_pretrained(CUSTOM_MODEL_DIR)
                    self._custom_model.eval()

                inputs = self._custom_tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
                with torch.no_grad():
                    outputs = self._custom_model.generate(**inputs, max_length=128, num_beams=4)
                translated = self._custom_tokenizer.decode(outputs[0], skip_special_tokens=True)
                return {
                    "translated_text": translated,
                    "src_lang": src_lang,
                    "tgt_lang": tgt_lang,
                    "mode": "custom_seq2seq_iitb",
                    "status": "success"
                }
            except Exception:
                pass

        # 2. Fall back to online NMT
        try:
            translator = GoogleTranslator(source=src_lang, target=tgt_lang)
            translated = translator.translate(text)
            return {
                "translated_text": translated,
                "src_lang": src_lang,
                "tgt_lang": tgt_lang,
                "mode": "online",
                "status": "success"
            }
        except Exception as e:
            # Fall back to offline matching if network fails
            fallback = self.match_offline(text, target_lang=tgt_lang)
            fallback["warning"] = f"Online translation unavailable ({str(e)}). Used offline fallback."
            return fallback

    def match_offline(self, text: str, target_lang: str = "hi", threshold: float = 0.22) -> dict:
        """
        Fuzzy matches input text against pre-indexed offline phrasebook.
        Returns closest translation along with phonetic transliteration for vendor communication.
        """
        best_match = None
        best_score = -1.0

        for item in self.phrases:
            # Match against English, phonetic, or Hindi
            score_eng = string_similarity(text, item["english"])
            score_jac = jaccard_similarity(text, item["english"])
            combined_score = (0.6 * score_eng) + (0.4 * score_jac)

            if combined_score > best_score:
                best_score = combined_score
                best_match = item

        lang_key_map = {
            "hi": "hindi",
            "en": "english",
            "es": "spanish",
            "fr": "french",
            "de": "german",
            "ja": "japanese"
        }
        target_key = lang_key_map.get(target_lang, "hindi")

        if best_match and best_score >= threshold:
            translated_text = best_match.get(target_key, best_match.get("hindi", ""))
            return {
                "translated_text": translated_text,
                "matched_source_phrase": best_match["english"],
                "category": best_match.get("category", "General"),
                "phonetic_transliteration": best_match.get("phonetic_transliteration", ""),
                "match_confidence": round(best_score, 3),
                "mode": "offline_phrasebook",
                "status": "matched"
            }
        else:
            # Generic fallback
            return {
                "translated_text": f"[Offline] {text}",
                "phonetic_transliteration": "",
                "match_confidence": 0.0,
                "mode": "offline_phrasebook",
                "status": "unmatched_generic"
            }

    def translate(self, text: str, src_lang: str = "en", tgt_lang: str = "hi", force_offline: bool = False) -> dict:
        """
        Unified translation method respecting offline toggle.
        """
        if force_offline:
            return self.match_offline(text, target_lang=tgt_lang)
        return self.translate_online(text, src_lang=src_lang, tgt_lang=tgt_lang)

    def get_phrasebook_by_category(self, category: str = "All") -> list[dict]:
        """
        Returns phrasebook entries filtered by category.
        """
        if category == "All":
            return self.phrases
        return [p for p in self.phrases if p.get("category") == category]


if __name__ == "__main__":
    t = MultilingualTranslator()
    print("--- Online Translation Test ---")
    res_online = t.translate("How much does this cost?", src_lang="en", tgt_lang="hi")
    print(f"Online (en -> hi): {res_online['translated_text']}")

    print("\n--- Offline Fuzzy Match Test ---")
    test_queries = [
        "give me some discount please",
        "how to get to train station",
        "make food non spicy",
        "call police immediately"
    ]
    for q in test_queries:
        res_offline = t.translate(q, tgt_lang="hi", force_offline=True)
        print(f"Query: '{q}' -> Matched: '{res_offline.get('matched_source_phrase')}'")
        print(f"  Hindi: {res_offline['translated_text']} | Pronunciation: {res_offline.get('phonetic_transliteration')} (Conf: {res_offline.get('match_confidence')})")
