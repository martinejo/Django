import unittest

from src.anomaly_web.config import get_model_spec, merge_hyperparameters


class ConfigTests(unittest.TestCase):
    def test_get_model_spec_ok(self):
        spec = get_model_spec("random_forest")
        self.assertEqual(spec.key, "random_forest")

    def test_get_model_spec_invalid(self):
        with self.assertRaises(ValueError):
            get_model_spec("no_existe")

    def test_merge_hyperparameters_override(self):
        params = merge_hyperparameters("gradient_boosting", {"n_estimators": 10})
        self.assertEqual(params["n_estimators"], 10)
        self.assertEqual(params["random_state"], 42)


if __name__ == "__main__":
    unittest.main()
