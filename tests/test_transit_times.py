import pytest
from ..src.transit_times import TransitTimes

class TestTransitTimes:
    """
    Tests:
    ** s = successful, us = unsuccessful
        test s that each variable is of np.ndarray type
        test us that each variable is of np.ndarray type
        test s that values in each array are of specified type (epochs=ints, mid_transit_times=floats, uncertainties=floats)
        test us that values in each array are of specified type (epochs=ints, mid_transit_times=floats, uncertainties=floats)
        test s that all variables have same shape
        test us that all variables have same shape
        test s that there are no null/nan values
        test us that there are no null/nan values
        test s that uncertainties are all non-negative and non-zero
        test s creation of uncertainties if not given
    """
    def test_all_data_success():
        pass

    def test_all_data_fail():
        pass

    def test_no_uncertainties():
        pass