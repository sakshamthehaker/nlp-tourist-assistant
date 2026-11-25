"""
test_app.py
Automated End-to-End API and Model Validation Test.
"""

import unittest
from app import app


class TestTourLingoApp(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_home_page(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"TourLingo AI", response.data)

    def test_chat_api(self):
        payload = {"query": "What time does Taj Mahal open?", "user_lang": "en"}
        response = self.client.post('/api/chat', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("answer", data)
        self.assertEqual(data["intent"], "ask_attraction")
        self.assertIn("Taj Mahal", data["answer"])

    def test_translate_api_offline(self):
        payload = {
            "text": "How much does this cost?",
            "src_lang": "en",
            "tgt_lang": "hi",
            "offline": True
        }
        response = self.client.post('/api/translate', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("translated_text", data)
        self.assertEqual(data["mode"], "offline_phrasebook")

    def test_recommend_api(self):
        payload = {"city": "Jaipur", "category": "All"}
        response = self.client.post('/api/recommend', json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(len(data["recommendations"]) > 0)
        self.assertEqual(data["city"], "Jaipur")

    def test_phrases_api(self):
        response = self.client.get('/api/phrases?category=Bargaining%20%26%20Shopping')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertTrue(len(data["phrases"]) > 0)

    def test_metrics_api(self):
        response = self.client.get('/api/metrics')
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn("project_id", data)
        self.assertIn("machine_translation", data)


if __name__ == '__main__':
    unittest.main()
