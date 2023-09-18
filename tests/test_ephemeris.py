import pytest
from unittest.mock import Mock
from pytest_mock import mocker
from ..src.ephemeris import Ephemeris

class TestEphemeris:
    def test_transit_obj_creation(self, mocker):
        mocker.patch.object(Ephemeris, )
        ephemeris = Ephemeris()

    def test_get_model_parameters_linear(self):
        ephemeris = Ephemeris()
        pass

    def test_get_model_parameters_quad():
        pass