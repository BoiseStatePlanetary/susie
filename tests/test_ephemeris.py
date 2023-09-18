import pytest
from unittest.mock import Mock
from pytest_mock import mocker
from ..src.ephemeris import Ephemeris

class TestEphemeris:
    """
    Tests:
        s initialization of object (given correct params)
        us initialization of object (given incorrect params, none, or too many)
        s method call of get_model_parameters (linear & quad)
        u method call of get_model_parameters (linear & quad)

    """
    def test_get_model_parameters_linear(self):
        ephemeris = Ephemeris()
        pass

    def test_get_model_parameters_quad():
        pass