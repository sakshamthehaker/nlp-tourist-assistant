"""
recommender.py
Location-based & Preference-aware Recommendation Engine for Tourists.
Uses the Haversine formula for geodesic distance and a multi-attribute utility ranking model.
"""

import os
import math
import json

KB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "destination_knowledge.json")

# Default city coordinates (city centers)
CITY_COORDINATES = {
    "Delhi": (28.6139, 77.2090),
    "Agra": (27.1767, 78.0081),
    "Jaipur": (26.9124, 75.7873),
    "Varanasi": (25.3176, 82.9739)
}


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes the great-circle distance in kilometers between two GPS coordinates
    using the Haversine formula.
    """
    R = 6371.0 # Earth's radius in kilometers
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


class TouristRecommender:
    def __init__(self, kb_path=KB_PATH):
        with open(kb_path, "r", encoding="utf-8") as f:
            self.destinations = json.load(f)["destinations"]

    def recommend(
        self,
        user_lat: float = None,
        user_lon: float = None,
        city: str = None,
        category: str = "All",
        max_distance_km: float = 30.0,
        min_rating: float = 4.0,
        top_n: int = 5
    ) -> list[dict]:
        """
        Ranks and recommends places based on location proximity, category, and quality ratings.
        """
        # Resolve coordinates from city if not provided
        if (user_lat is None or user_lon is None) and city:
            city_norm = city.capitalize()
            if city_norm in CITY_COORDINATES:
                user_lat, user_lon = CITY_COORDINATES[city_norm]
            else:
                user_lat, user_lon = CITY_COORDINATES["Delhi"]

        if user_lat is None or user_lon is None:
            user_lat, user_lon = CITY_COORDINATES["Delhi"]

        candidates = []
        for dest in self.destinations:
            # City filter (if city is selected)
            if city and dest["city"].lower() != "pan-india" and dest["city"].lower() != city.lower():
                continue

            # Category filter
            if category != "All" and dest["category"] != category:
                continue

            # Rating filter
            rating = dest.get("rating", 4.5)
            if rating < min_rating:
                continue

            # Distance calculation
            dest_lat = dest.get("lat", user_lat)
            dest_lon = dest.get("lon", user_lon)
            dist_km = haversine_distance(user_lat, user_lon, dest_lat, dest_lon)

            # Combined ranking score (Proximity + Quality)
            proximity_score = 1.0 / (1.0 + (dist_km * 0.15))
            quality_score = rating / 5.0
            overall_score = round((0.45 * proximity_score) + (0.55 * quality_score), 4)

            item = dict(dest)
            item["distance_km"] = dist_km
            item["relevance_score"] = overall_score
            candidates.append(item)

        # Sort by relevance_score descending
        candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
        return candidates[:top_n]


if __name__ == "__main__":
    recommender = TouristRecommender()
    print("--- Top recommendations near Jaipur City Center ---")
    recs = recommender.recommend(city="Jaipur", category="All", top_n=3)
    for r in recs:
        print(f"• {r['name']} ({r['category']}) - {r['distance_km']} km away | Rating: {r['rating']}⭐ | Score: {r['relevance_score']}")
