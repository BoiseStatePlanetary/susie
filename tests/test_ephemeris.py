import sys
sys.path.append(".")
import unittest
from src.susie.ephemeris import Ephemeris

class TestEphemeris(unittest.TestCase):
    """
    Tests:
        s initialization of object (given correct params)
        us initialization of object (given incorrect params, none, or too many)
        s method call of get_model_parameters (linear & quad)
        u method call of get_model_parameters (linear & quad)

    """
    def test_get_model_parameters_linear(self):
        # ephemeris = Ephemeris()
        pass

    def test_get_model_parameters_quad(self):
        pass