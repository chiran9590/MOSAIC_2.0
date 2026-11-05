import unittest

from src.strategy.risk_manager import RiskConfig, RiskManager


class TestRiskManager(unittest.TestCase):

    def setUp(self):
        self.manager = RiskManager()

    def test_long_trade(self):
        result = self.manager.assess_trade("LONG", 1500)
        self.assertTrue(result.accepted)
        self.assertEqual(result.stop_loss, 1485)
        self.assertEqual(result.quantity, 13)

    def test_short_trade(self):
        result = self.manager.assess_trade("SHORT", 1500)
        self.assertTrue(result.accepted)
        self.assertEqual(result.stop_loss, 1515)

    def test_position_value_limit(self):
        result = self.manager.assess_trade("LONG", 1500)
        self.assertLessEqual(result.position_value, 20000)

    def test_total_risk_within_budget(self):
        result = self.manager.assess_trade("LONG", 1500)
        self.assertLessEqual(result.estimated_total_risk, 500)

    def test_invalid_direction(self):
        with self.assertRaises(ValueError):
            self.manager.assess_trade("UP", 1500)

    def test_invalid_entry_price(self):
        with self.assertRaises(ValueError):
            self.manager.assess_trade("LONG", 0)

    def test_invalid_configuration(self):
        with self.assertRaises(ValueError):
            RiskManager(RiskConfig(capital=-1000))


if __name__ == "__main__":
    unittest.main(verbosity=2)
