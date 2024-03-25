import sys
sys.path.append(".")
import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.susie.transit_times import TransitTimes
from src.susie.ephemeris import Ephemeris
from src.susie.ephemeris import LinearModelEphemeris
test_epochs = np.array([0, 294, 298, 573])
test_mtts = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
test_mtts_err = np.array([0.00043, 0.00028, 0.00062, 0.00042])
test_x=np.array([631.933999999892, 320.8780000000261, 325.24399999994785, 625.3850000002421])
test_P= 1.0914196486248962
test_T0= 3.0
       


class TestLinearModelEphemeris(unittest.TestCase):
    def linear_fit_instantiation(self):
        self.assertIsInstance(self.emphemeris, LinearModelEphemeris)
        self.linearmodelemphemeris= LinearModelEphemeris() 
    


    ##not working but the arrays are equal
    ## says x is the mid transit times but are ints which doesnt line up
    def test_linear_fit(self):
        linear_model=LinearModelEphemeris()
        expected_result=np.array([692.70518423 , 353.21255401, 357.9776922, 685.55747696])
        result= linear_model.lin_fit(test_x, test_P, test_T0)
        print("Expected Result:", expected_result)
        print("Actual Result:", result)
        self.assertTrue(np.array_equal(expected_result,result))

  
class TestEphemeris(unittest.TestCase):
    """
    Tests:
        s initialization of object (given correct params)
        us initialization of object (given incorrect params, none, or too many)
        s method call of get_model_parameters (linear & quad)
        u method call of get_model_parameters (linear & quad)

    """
    def setUp(self):
       self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
    
    def test_get_model_parameters_linear(self):
        # ephemeris = Ephemeris()
        pass

    def test_get_model_parameters_quad(self):
        pass


if __name__ == '__main__':
    unittest.main()