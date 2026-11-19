import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import unittest
import numpy as np
from fastapi.testclient import TestClient
from src.model import OutcomePredictor
from src.explainability import ModelExplainer
import main
from main import app


class TestBARBARIKModelAndAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Initialize client and model
        cls.client = TestClient(app)
        main.initialize_model()
        cls.predictor = main.predictor
        cls.explainer = main.explainer

    def test_01_model_loaded(self):
        """Verify model, preprocessor, and label encoder are loaded."""
        self.assertIsNotNone(self.predictor)
        self.assertTrue(self.predictor.is_trained)
        self.assertTrue(hasattr(self.predictor, 'label_encoder'))
        self.assertEqual(len(self.predictor.label_encoder.classes_), 3)

    def test_02_predict_blue_advantage(self):
        """Test scenario heavily favoring Side A (Blue Force)."""
        scenario_blue = {
            "side_a_force_strength": 95,
            "side_b_force_strength": 35,
            "side_a_mobility": 0.95,
            "side_b_mobility": 0.35,
            "side_a_logistics": 0.95,
            "side_b_logistics": 0.35,
            "side_a_intelligence": 0.95,
            "side_b_intelligence": 0.35,
            "side_a_communication": 0.95,
            "side_b_communication": 0.35,
            "side_a_readiness": 0.95,
            "side_b_readiness": 0.35,
            "side_a_morale": 0.95,
            "side_b_morale": 0.25,
            "side_a_training": 0.95,
            "side_b_training": 0.35,
            "side_a_supply_distance": 20,
            "side_b_supply_distance": 450,
            "side_a_surprise": 0.9,
            "side_b_surprise": 0.1,
            "side_a_fatigue": 0.1,
            "side_b_fatigue": 0.9,
            "side_a_equipment_quality": 0.95,
            "side_b_equipment_quality": 0.35,
            "electronic_warfare_level": 0.0,
            "objective_type": "Attack",
            "season": "Summer",
            "rules_of_engagement": "Offensive",
            "terrain_type": "Open",
            "weather": "Clear",
            "visibility": "High",
            "time_of_day": "Day",
            "environmental_difficulty": 0.1
        }
        res = self.client.post("/api/predict", json=scenario_blue)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_class"], "SIDE_A_ADVANTAGE")
        self.assertIn("BLUE", data["outcome"])
        self.assertGreater(data["probabilities"]["SIDE_A_ADVANTAGE"], 0.5)

    def test_03_predict_red_advantage(self):
        """Test scenario heavily favoring Side B (Red Force)."""
        scenario_red = {
            "side_a_force_strength": 35,
            "side_b_force_strength": 95,
            "side_a_mobility": 0.35,
            "side_b_mobility": 0.95,
            "side_a_logistics": 0.35,
            "side_b_logistics": 0.95,
            "side_a_intelligence": 0.35,
            "side_b_intelligence": 0.95,
            "side_a_communication": 0.35,
            "side_b_communication": 0.95,
            "side_a_readiness": 0.35,
            "side_b_readiness": 0.95,
            "side_a_morale": 0.25,
            "side_b_morale": 0.95,
            "side_a_training": 0.35,
            "side_b_training": 0.95,
            "side_a_supply_distance": 450,
            "side_b_supply_distance": 20,
            "side_a_surprise": 0.1,
            "side_b_surprise": 0.9,
            "side_a_fatigue": 0.9,
            "side_b_fatigue": 0.1,
            "side_a_equipment_quality": 0.35,
            "side_b_equipment_quality": 0.95,
            "electronic_warfare_level": 0.0,
            "objective_type": "Defend",
            "season": "Summer",
            "rules_of_engagement": "Defensive",
            "terrain_type": "Urban",
            "weather": "Clear",
            "visibility": "High",
            "time_of_day": "Day",
            "environmental_difficulty": 0.1
        }
        res = self.client.post("/api/predict", json=scenario_red)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_class"], "SIDE_B_ADVANTAGE")
        self.assertIn("RED", data["outcome"])
        self.assertGreater(data["probabilities"]["SIDE_B_ADVANTAGE"], 0.5)

    def test_04_predict_stalemate(self):
        """Test evenly matched scenario resulting in Stalemate."""
        scenario_stalemate = {
            "side_a_force_strength": 65,
            "side_b_force_strength": 65,
            "side_a_mobility": 0.65,
            "side_b_mobility": 0.65,
            "side_a_logistics": 0.65,
            "side_b_logistics": 0.65,
            "side_a_intelligence": 0.65,
            "side_b_intelligence": 0.65,
            "side_a_communication": 0.65,
            "side_b_communication": 0.65,
            "side_a_readiness": 0.65,
            "side_b_readiness": 0.65,
            "side_a_morale": 0.65,
            "side_b_morale": 0.65,
            "side_a_training": 0.65,
            "side_b_training": 0.65,
            "side_a_supply_distance": 100,
            "side_b_supply_distance": 100,
            "side_a_surprise": 0.2,
            "side_b_surprise": 0.2,
            "side_a_fatigue": 0.4,
            "side_b_fatigue": 0.4,
            "side_a_equipment_quality": 0.65,
            "side_b_equipment_quality": 0.65,
            "electronic_warfare_level": 0.0,
            "objective_type": "Delay",
            "season": "Summer",
            "rules_of_engagement": "Permissive",
            "terrain_type": "Mixed",
            "weather": "Clear",
            "visibility": "High",
            "time_of_day": "Day",
            "environmental_difficulty": 0.2
        }
        res = self.client.post("/api/predict", json=scenario_stalemate)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["predicted_class"], "STALEMATE")
        self.assertEqual(data["outcome"], "STALEMATE")
        self.assertGreater(data["probabilities"]["STALEMATE"], 0.4)

    def test_05_predict_contract_format(self):
        """Verify API response keys, probability sum, and SHAP features."""
        scenario = {
            "side_a_force_strength": 80,
            "side_b_force_strength": 70,
            "side_a_mobility": 0.75,
            "side_b_mobility": 0.60,
            "side_a_logistics": 0.70,
            "side_b_logistics": 0.55,
            "side_a_intelligence": 0.80,
            "side_b_intelligence": 0.60,
            "side_a_communication": 0.85,
            "side_b_communication": 0.65,
            "side_a_readiness": 0.75,
            "side_b_readiness": 0.60,
            "side_a_morale": 0.8,
            "side_b_morale": 0.6,
            "side_a_training": 0.75,
            "side_b_training": 0.55,
            "side_a_supply_distance": 80,
            "side_b_supply_distance": 150,
            "side_a_surprise": 0.6,
            "side_b_surprise": 0.3,
            "side_a_fatigue": 0.2,
            "side_b_fatigue": 0.4,
            "side_a_equipment_quality": 0.8,
            "side_b_equipment_quality": 0.6,
            "electronic_warfare_level": 0.25,
            "objective_type": "Attack",
            "season": "Summer",
            "rules_of_engagement": "Permissive",
            "terrain_type": "Hilly",
            "weather": "Cloudy",
            "visibility": "Medium",
            "time_of_day": "Day",
            "environmental_difficulty": 0.3
        }
        res = self.client.post("/api/predict", json=scenario)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        
        # Required top-level keys
        for key in ["outcome", "predicted_class", "probabilities", "confidence", "simulation_metrics", "shap_values"]:
            self.assertIn(key, data)
        
        # Probability structure and sum
        probs = data["probabilities"]
        for cls_name in ["SIDE_A_ADVANTAGE", "SIDE_B_ADVANTAGE", "STALEMATE"]:
            self.assertIn(cls_name, probs)
            self.assertIsInstance(probs[cls_name], float)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=2)
        
        # Simulation metrics
        sim = data["simulation_metrics"]
        for key in ["duration", "score_a", "score_b", "resource_status_a", "resource_status_b", "casualties_blue", "casualties_red"]:
            self.assertIn(key, sim)
            
        # SHAP values
        shap_vals = data["shap_values"]
        self.assertIsInstance(shap_vals, list)
        self.assertGreater(len(shap_vals), 0)
        for item in shap_vals:
            self.assertIn("feature", item)
            self.assertIn("value", item)
            self.assertIsInstance(item["value"], float)

    def test_06_missing_and_invalid_inputs(self):
        """Test API validation handles missing, empty, or invalid type inputs."""
        # Empty body
        res_empty = self.client.post("/api/predict", json={})
        self.assertEqual(res_empty.status_code, 422)
        
        # Missing required parameter (e.g. side_a_force_strength)
        incomplete = {
            "side_b_force_strength": 70,
            "side_a_mobility": 0.75
        }
        res_incomplete = self.client.post("/api/predict", json=incomplete)
        self.assertEqual(res_incomplete.status_code, 422)
        
        # Invalid data type (string instead of int/float)
        invalid_type = {
            "side_a_force_strength": "invalid_number_string",
            "side_b_force_strength": 70,
            "side_a_mobility": 0.75,
            "side_b_mobility": 0.60,
            "side_a_logistics": 0.70,
            "side_b_logistics": 0.55,
            "side_a_intelligence": 0.80,
            "side_b_intelligence": 0.60,
            "side_a_communication": 0.85,
            "side_b_communication": 0.65,
            "side_a_readiness": 0.75,
            "side_b_readiness": 0.60,
            "side_a_morale": 0.8,
            "side_b_morale": 0.6,
            "side_a_training": 0.75,
            "side_b_training": 0.55,
            "side_a_supply_distance": 80,
            "side_b_supply_distance": 150,
            "side_a_surprise": 0.6,
            "side_b_surprise": 0.3,
            "side_a_fatigue": 0.2,
            "side_b_fatigue": 0.4,
            "side_a_equipment_quality": 0.8,
            "side_b_equipment_quality": 0.6,
            "electronic_warfare_level": 0.25,
            "objective_type": "Attack",
            "season": "Summer",
            "rules_of_engagement": "Permissive",
            "terrain_type": "Hilly",
            "weather": "Cloudy",
            "visibility": "Medium",
            "time_of_day": "Day",
            "environmental_difficulty": 0.3
        }
        res_invalid = self.client.post("/api/predict", json=invalid_type)
        self.assertEqual(res_invalid.status_code, 422)

    def test_07_metrics_endpoint(self):
        """Test GET /api/model/metrics returns expected metrics with >90% accuracy."""
        res = self.client.get("/api/model/metrics")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        for key in ["accuracy", "precision", "recall", "f1", "roc_auc", "confusion_matrix", "roc_curve_data", "feature_importances"]:
            self.assertIn(key, data)
        self.assertGreaterEqual(data["accuracy"], 0.90)
        self.assertGreaterEqual(data["roc_auc"], 0.95)
        self.assertEqual(len(data["confusion_matrix"]), 3)

    def test_08_history_endpoint(self):
        """Test GET /api/history returns valid history list."""
        res = self.client.get("/api/history")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("history", data)
        self.assertIsInstance(data["history"], list)


if __name__ == '__main__':
    unittest.main()
