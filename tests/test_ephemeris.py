import sys
sys.path.append(".")
import unittest
import numpy as np
import matplotlib.pyplot as plt
from src.susie.transit_times import TransitTimes
from src.susie.ephemeris import Ephemeris, LinearModelEphemeris
from scipy.optimize import curve_fit
test_epochs = np.array([0, 294, 298, 573])
test_mtts = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
test_mtts_err = np.array([0.00043, 0.00028, 0.00062, 0.00042])
test_x = np.array([631.933999999892, 320.8780000000261, 325.24399999994785, 625.3850000002421])
test_P = 1.0914196486248962
test_T0 = 3.0

print(type(test_x[0]))
       


class TestLinearModelEphemeris(unittest.TestCase):
    def linear_fit_instantiation(self):
        self.ephemeris = LinearModelEphemeris()
        self.assertIsInstance(self.emphemeris, LinearModelEphemeris)
    
    def linear_fit_parameters(self):
        pass

    ##not working but the arrays are equal
    ## says x is the mid transit times but are ints which doesnt line up
    def test_linear_fit(self):
        linear_model = LinearModelEphemeris()
        expected_result = np.array([692.70518423, 353.21255401, 357.9776922, 685.55747696])
        result = linear_model.lin_fit(test_x, test_P, test_T0)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_fit_model(self):
        linear_model = LinearModelEphemeris()
        popt, pcov = curve_fit(linear_model.lin_fit, test_epochs, test_mtts, sigma=test_mtts_err, absolute_sigma=True)
        unc = np.sqrt(np.diag(pcov))
        print(popt)
        print(pcov)
        print(unc)
        return_data = {
            'period': popt[0],
            'period_err': unc[0],
            'conjunction_time': popt[1],
            'conjunction_time_err': unc[1]
        }
        self.assertEqual(popt[0], return_data['period'])
        # do with conjunction time as well

  
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