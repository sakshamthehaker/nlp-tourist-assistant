"""
demo_scenarios.py
Demonstrates the 6 representative tourist scenarios specified in project requirements:
1. Attraction & monument timings Q&A
2. Local public transport & metro guidance
3. Cultural customs & temple etiquette
4. Real-time tourist-vendor bargaining (with phonetic transliteration)
5. Location-based food & service recommendation
6. Emergency SOS and offline fallback operation
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.intent_classifier import TouristIntentClassifier
from src.semantic_retriever import DestinationRetriever
from src.translator import MultilingualTranslator
from src.recommender import TouristRecommender


def run_demonstration():
    print("=" * 75)
    print("TourLingo AI - REPRESENTATIVE TOURIST SCENARIOS DEMONSTRATION")
    print("Project ID: CSEAIML2026-P26 | NLP Course Project")
    print("=" * 75)

    clf = TouristIntentClassifier.load()
    retriever = DestinationRetriever()
    translator = MultilingualTranslator()
    recommender = TouristRecommender()

    scenarios = [
        {
            "num": 1,
            "title": "Historical Attraction & Ticket Query",
            "tourist_query": "What time does Taj Mahal open and is it closed on Fridays?",
            "lang": "en"
        },
        {
            "num": 2,
            "title": "Public Transport & Metro Navigation",
            "tourist_query": "How do I take the Delhi Metro from the airport to Red Fort?",
            "lang": "en"
        },
        {
            "num": 3,
            "title": "Cultural Customs & Religious Etiquette",
            "tourist_query": "What should I wear when visiting Hindu temples in India?",
            "lang": "en"
        },
        {
            "num": 4,
            "title": "Real-Time Tourist-Vendor Bargaining (Live Translation)",
            "tourist_query": "This is too expensive. Can you give a discount?",
            "action": "translate",
            "src": "en",
            "tgt": "hi"
        },
        {
            "num": 5,
            "title": "Location-Based Nearby Recommendation (Jaipur Food & Culture)",
            "action": "recommend",
            "city": "Jaipur",
            "category": "Food & Dining"
        },
        {
            "num": 6,
            "title": "Offline-Friendly Emergency SOS Assistance",
            "tourist_query": "I need help immediately! Call the police.",
            "action": "translate_offline",
            "tgt": "hi"
        }
    ]

    for sc in scenarios:
        print(f"\n[SCENARIO {sc['num']}]: {sc['title']}")
        print("-" * 65)

        if sc.get("action") == "translate":
            res = translator.translate(sc["tourist_query"], src_lang=sc["src"], tgt_lang=sc["tgt"])
            print(f"Tourist (English): '{sc['tourist_query']}'")
            print(f"Vendor Sees (Hindi):  {res['translated_text']}")
            if res.get("phonetic_transliteration"):
                print(f"Tourist Pronounces:   {res['phonetic_transliteration']}")
            print(f"Translation Mode:     {res.get('mode', 'online')}")

        elif sc.get("action") == "translate_offline":
            res = translator.translate(sc["tourist_query"], tgt_lang=sc["tgt"], force_offline=True)
            print(f"Tourist Query:        '{sc['tourist_query']}' (Offline Network Mode)")
            print(f"Matched Phrase:       '{res.get('matched_source_phrase')}'")
            print(f"Vendor Sees (Hindi):  {res['translated_text']}")
            print(f"Pronunciation Guide:  {res.get('phonetic_transliteration')}")
            print(f"Match Confidence:     {res.get('match_confidence')}")

        elif sc.get("action") == "recommend":
            recs = recommender.recommend(city=sc["city"], category=sc["category"], top_n=2)
            print(f"User Location: {sc['city']} | Filter: {sc['category']}")
            for r in recs:
                print(f"• {r['name']} ({r['city']}) - {r['distance_km']} km away | Rating: {r['rating']}⭐")
                print(f"  Price: {r['price_level']}")

        else: # Q&A
            pred = clf.predict(sc["tourist_query"])
            qa = retriever.answer_query(sc["tourist_query"], pred)
            print(f"Tourist Query: '{sc['tourist_query']}'")
            print(f"NLP Predicted Intent: {pred['intent']} (Confidence: {pred['confidence'] * 100:.1f}%)")
            print(f"Extracted Entities:   {pred['entities']}")
            print(f"Assistant Response:\n{qa['answer']}")

    print("\n" + "=" * 75)
    print("ALL 6 REPRESENTATIVE SCENARIOS SUCCESSFULLY DEMONSTRATED!")
    print("=" * 75)


if __name__ == "__main__":
    run_demonstration()
