"""
semantic_retriever.py
Information Retrieval & Semantic Question Answering over Destination Knowledge Base.
Uses Granular Passage Chunking + TF-IDF Cosine Similarity for pinpoint factual retrieval.
"""

import os
import sys
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import preprocess

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "destination_knowledge.json")


class DestinationRetriever:
    def __init__(self, kb_path=KB_PATH):
        self.kb_path = kb_path
        self.destinations = []
        self.chunks = []
        self.vectorizer = None
        self.chunk_vectors = None
        self.load_and_index()

    def load_and_index(self):
        with open(self.kb_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.destinations = data["destinations"]

        self.chunks = []
        for dest in self.destinations:
            name = dest["name"]
            city = dest["city"]

            # 1. Overview Chunk
            self.chunks.append({
                "destination": dest,
                "facet": "overview",
                "text": f"{name} {city} {dest['category']} {dest['type']} {dest['description']}"
            })

            # 2. Timing & Pricing Chunk
            if dest.get("opening_hours") or dest.get("price_level"):
                self.chunks.append({
                    "destination": dest,
                    "facet": "timing_pricing",
                    "text": f"{name} {city} opening hours timings ticket fee cost price admission {dest.get('opening_hours', '')} {dest.get('price_level', '')}"
                })

            # 3. Transport Chunk
            if dest.get("transport"):
                self.chunks.append({
                    "destination": dest,
                    "facet": "transport",
                    "text": f"{name} {city} transport metro train bus taxi auto rickshaw route reach travel directions {dest['transport']}"
                })

            # 4. Etiquette Chunk
            if dest.get("etiquette"):
                self.chunks.append({
                    "destination": dest,
                    "facet": "etiquette",
                    "text": f"{name} {city} etiquette customs dress code rules shoes culture temple manners tips {dest['etiquette']}"
                })

            # 5. Emergency Chunk
            if dest.get("emergency"):
                self.chunks.append({
                    "destination": dest,
                    "facet": "emergency",
                    "text": f"{name} {city} emergency police hospital doctor helpline assistance lost passport sos {dest['emergency']}"
                })

        corpus = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            preprocessor=preprocess,
            ngram_range=(1, 3), # Unigram, bigram, trigram
            sublinear_tf=True
        )
        self.chunk_vectors = self.vectorizer.fit_transform(corpus)

    def retrieve(self, query: str, top_k: int = 3, threshold: float = 0.05) -> list[dict]:
        """
        Retrieves top_k most relevant passage chunks using Cosine Similarity.
        """
        q_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(q_vec, self.chunk_vectors)[0]
        
        ranked_indices = np.argsort(similarities)[::-1]
        results = []
        seen_dest_ids = set()

        for idx in ranked_indices:
            sim = float(similarities[idx])
            if sim < threshold and len(results) > 0:
                break
            chunk = self.chunks[idx]
            dest_id = chunk["destination"]["id"]
            
            # Return unique top destinations with matching chunk facet
            if dest_id not in seen_dest_ids or len(seen_dest_ids) < top_k:
                seen_dest_ids.add(dest_id)
                res = dict(chunk["destination"])
                res["matched_facet"] = chunk["facet"]
                res["similarity_score"] = round(sim, 4)
                results.append(res)
                if len(results) >= top_k:
                    break

        return results

    def answer_query(self, query: str, intent_info: dict) -> dict:
        """
        Synthesizes an intelligent answer based on granular passage chunks and intent.
        """
        intent = intent_info.get("intent", "ask_attraction")
        entities = intent_info.get("entities", {})

        target_poi = entities.get("poi")
        matched_dest = None

        if target_poi:
            for d in self.destinations:
                if target_poi.lower() in d["name"].lower() or d["name"].lower() in target_poi.lower():
                    matched_dest = d
                    break

        retrieved = self.retrieve(query, top_k=2)
        top_dest = matched_dest if matched_dest else (retrieved[0] if retrieved else self.destinations[0])

        if intent == "general_greeting":
            reply = (
                "Hello! I am your AI Multilingual Tourist Guide. "
                "I can help you explore top attractions, navigate local transport, recommend food, "
                "share cultural etiquette & customs, or provide emergency SOS assistance. "
                "How can I assist your travels today?"
            )
        elif intent == "emergency_sos":
            reply = (
                f"🚨 **Emergency & Safety Assistance:**\n"
                f"• **National Emergency Helpline:** Call **112** (Police, Ambulance, Fire)\n"
                f"• **Tourist Police Multilingual Helpline:** Call **1363** (24/7 Toll-free)\n"
                f"• **Local Assistance Info:** {top_dest.get('emergency', 'Contact nearest police kiosk or dial 112.')}\n\n"
                f"💡 *Tip:* If you lost belongings or your passport, request a stamped Non-Cognizable Report (NCR) from the nearest Tourist Police booth."
            )
        elif intent == "ask_transport":
            reply = (
                f"🚆 **Transport Guidance for {top_dest['name']} ({top_dest['city']}):**\n\n"
                f"{top_dest['transport']}\n\n"
                f"• **Opening Hours:** {top_dest.get('opening_hours', 'N/A')}\n"
                f"• **Price/Fare:** {top_dest.get('price_level', 'N/A')}"
            )
        elif intent == "ask_etiquette":
            reply = (
                f"🙏 **Cultural Etiquette & Customs for {top_dest['name']} ({top_dest['city']}):**\n\n"
                f"{top_dest['etiquette']}\n\n"
                f"• **General Tip:** Always remove shoes before religious premises, dress modestly covering shoulders and knees, and use your right hand for offerings or payments."
            )
        elif intent == "food_recommendation":
            food_candidates = [d for d in self.destinations if d["category"] == "Food & Dining"]
            food_item = food_candidates[0] if food_candidates else top_dest
            reply = (
                f"🍽️ **Dining Recommendation - {food_item['name']} ({food_item['city']}):**\n\n"
                f"{food_item['description']}\n\n"
                f"• **Price Range:** {food_item.get('price_level', 'Budget-friendly')}\n"
                f"• **Operating Hours:** {food_item.get('opening_hours', '11:00 AM - 11:00 PM')}\n"
                f"• **Tips:** {food_item.get('etiquette', 'Packaged water recommended; inform vendor if you prefer low spice.')}"
            )
        elif intent == "currency_budget":
            reply = (
                f"💳 **Currency & Budget Advice:**\n\n"
                f"• **Official Currency:** Indian Rupee (INR ₹)\n"
                f"• **Foreign Travelers UPI:** International tourists can now get prepaid UPI wallets (like Cheq UPI / Ebix) at Delhi and major international airports.\n"
                f"• **Cards & Cash:** Credit/Debit cards are widely accepted at hotels and monument ticket counters ({top_dest.get('price_level', '')}). Street vendors and auto-rickshaws prefer cash or UPI.\n"
                f"• **ATMs:** Use bank-attached ATMs (SBI, HDFC, ICICI) to avoid card-skimming."
            )
        elif intent == "bargaining_vendor":
            reply = (
                f"🛍️ **Local Market Bargaining & Vendor Guide:**\n\n"
                f"• **Where to bargain:** Street markets (Janpath, Sarojini Nagar, Johari Bazaar, local souvenirs).\n"
                f"• **Golden Rule:** Politely start at 50-60% of the initial quoted price and settle with a warm smile around 70%.\n"
                f"• **Fixed Price Shops:** Government Emporiums (e.g., Central Cottage Industries) have fixed, authentic prices.\n"
                f"• **Handy Phrase:** *'Yeh kitne ka hai? Thoda kam kijiye please.'* (How much is this? Please give a little discount)."
            )
        else: # ask_attraction
            reply = (
                f"🏛️ **{top_dest['name']} ({top_dest['city']})**\n"
                f"*{top_dest['type']}* | ⭐ {top_dest.get('rating', '4.5')}/5.0\n\n"
                f"{top_dest['description']}\n\n"
                f"• **Timings:** {top_dest.get('opening_hours', 'Sunrise to Sunset')}\n"
                f"• **Ticket Fee:** {top_dest.get('price_level', 'Standard entry fee applies')}\n"
                f"• **How to Reach:** {top_dest['transport']}\n"
                f"• **Etiquette:** {top_dest['etiquette']}"
            )

        return {
            "answer": reply,
            "matched_destination": top_dest["name"],
            "city": top_dest["city"],
            "similarity": top_dest.get("similarity_score", 1.0)
        }


if __name__ == "__main__":
    retriever = DestinationRetriever()
    test_q = "What time does Taj Mahal open and is it closed on Fridays?"
    dummy_intent = {"intent": "ask_attraction", "entities": {"poi": "Taj Mahal"}}
    res = retriever.answer_query(test_q, dummy_intent)
    print("\n--- Granular Chunk Retrieval Test ---")
    print(res["answer"])
