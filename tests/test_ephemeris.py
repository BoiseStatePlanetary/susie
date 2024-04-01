import sys
sys.path.append(".")
import unittest
import numpy as np
import matplotlib.pyplot as plt
from susie.transit_times import TransitTimes
from src.susie.ephemeris import Ephemeris, LinearModelEphemeris, QuadraticModelEphemeris, ModelEphemerisFactory
from scipy.optimize import curve_fit
test_epochs = np.array([0, 294, 298, 573])
test_mtts = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
test_mtts_err = np.array([0.00043, 0.00028, 0.00062, 0.00042])
test_x = np.array([631.933999999892, 320.8780000000261, 325.24399999994785, 625.3850000002421])
test_P_linear = 1.0914196486248962
test_P_quad = 1.0914213749985644
test_dPdE = -8.367463423862439e-10
test_dPdE_err=7.510573769001014e-10
test_T0 = 3.0
test_T0_err_linear = 0.0007184099304791794
test_P_err_linear = 3.187601645504582e-07
test_T0_err_quad=0.0014642050906115362
test_P_err_quad=0.0014642050906115362
test_conjunction_time_quad =0.0016704341826483176
test_conjunction_time_err_quad =  0.0014642050906115362
test_epochs = np.array([0, 294, 298, 573])

print(type(test_x[0]))
       


class TestLinearModelEphemeris(unittest.TestCase):
    def linear_fit_instantiation(self):
        self.ephemeris = LinearModelEphemeris()
        self.assertIsInstance(self.emphemeris, LinearModelEphemeris)
    
    def test_linear_fit(self):
        linear_model = LinearModelEphemeris()
        expected_result = np.array([692.70518423, 353.21255401, 357.9776922, 685.55747696])
        result = linear_model.lin_fit(test_x, test_P_linear, test_T0)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_lin_fit_model(self):
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
        self.assertEqual(unc[0], return_data['period_err'])
        # do with conjunction time as well
        self.assertEqual(popt[1], return_data['conjunction_time'])
        self.assertEqual(unc[1], return_data['conjunction_time_err'])
class TestQuadraticModelEmphemeris(unittest.TestCase):

    def quad_fit_instantiation(self):
        self.ephemeris=QuadraticModelEphemeris()
        self.assertIsInstance(self.emphemeris, QuadraticModelEphemeris)

    def test_quad_fit(self):
        quadratic_model=QuadraticModelEphemeris()
        expected_result= np.array([692.70501716, 353.21251093, 357.97764794, 685.55731333])
        result = quadratic_model.quad_fit(test_x,test_dPdE, test_P_quad, test_T0)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_quad_fit_model(self):
        quad_model = QuadraticModelEphemeris()
        popt, pcov = curve_fit(quad_model.quad_fit, test_epochs, test_mtts, sigma=test_mtts_err, absolute_sigma=True)
        unc = np.sqrt(np.diag(pcov))
        print(popt)
        print(pcov)
        print(unc)
        return_data = {
            'conjunction_time': popt[2],
            'conjunction_time_err': unc[2],
            'period': popt[1],
            'period_err': unc[1],
            'period_change_by_epoch': popt[0],
            'period_change_by_epoch_err': unc[0],
        }
        self.assertEqual(popt[1], return_data['period'])
        self.assertEqual(unc[1], return_data['period_err'])
        self.assertEqual(popt[2], return_data['conjunction_time'])
        self.assertEqual(unc[2], return_data['conjunction_time_err'])
        self.assertEqual(popt[0], return_data['period_change_by_epoch'])
        self.assertEqual(unc[0], return_data['period_change_by_epoch_err'])


class TestModelEphemerisFactory(unittest.TestCase):
    def model_no_errors(self):
        models = {
            'linear': LinearModelEphemeris(),
            'quadratic': QuadraticModelEphemeris()
        }
        test_model_type= 'linear'
        self.assertTrue(test_model_type in models)
    
    def model_errors(self):
        models = {
            'linear': LinearModelEphemeris(),
            'quadratic': QuadraticModelEphemeris()
        }
        test_model_type= 'invaild_model'  
        with self.assertRaises(ValueError, msg=f"Invalid model type: {test_model_type}"):
            model = models[test_model_type]

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
       self.assertIsInstance(self.transit_times, TransitTimes)
       print('Beep')
       print(type(self.transit_times))
       self.ephemeris=Ephemeris(self.transit_times)
      

    # def transit_time_instantiation(self):
    #     self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
    #     self.assertIsInstance(self.transit_times, TransitTimes)
    #     pass

    # def test_us_transit_time_instantiation(self):
    #     with self.assertRaises(ValueError, msg="Variable 'transit_times' expected type of object 'TransitTimes'."):
    #         TransitTimes('jd', test_epochs, test_mtts_err, test_mtts_err, time_scale='tdb')
    #         self.transit_times=None
    #         self.ephemeris._validate()
    #         pass

    # def test_get_model_parameters_linear(self):
    #     test_model_type= 'linear'
    #     test_model_ephemeris_data = ModelEphemerisFactory().create_model(test_model_type,test_epochs, test_mtts, test_mtts_err, sigma=test_mtts_err, absolute_sigma=True)
    #     model_parameters = Ephemeris()._get_model_parameters(test_model_ephemeris_data)
    #     expected_result={
    #         'period': [1.0914196486248962],  
    #         'period_err': [1.5818372240135891e-06],  
    #         'conjunction_time': [ 0.0016704341826483176], 
    #         'conjunction_time_err': [0.0014642050906115362] 
    #     }
    #     self.assertEqual(model_parameters,expected_result)   
    #     pass

    # def test_get_model_parameters_quad(self):
    #     pass

    def test_k_value_linear(self):
        test_model_type='linear'
        expected_result= 2
        result=Ephemeris._get_k_value(self,test_model_type)
        self.assertEqual(result,expected_result)
    
    def test_k_value_quad(self):
        test_model_type='quadratic'
        expected_result= 3
        result=Ephemeris._get_k_value(self,test_model_type)
        self.assertEqual(result,expected_result)

    
    def test_calc_linear_model_uncertainties(self):
        expected_result=np.array([0.00071841, 0.0007245 , 0.00072466, 0.00074126])
        result=Ephemeris._calc_linear_model_uncertainties(self, test_T0_err_linear, test_P_err_linear)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_quad_model_uncertainties(self):
        expected_result=np.array([0.00146421, 0.43047879, 0.43633557, 0.8389908 ])
        result=Ephemeris._calc_quadratic_model_uncertainties(self, test_T0_err_quad, test_P_err_quad,test_dPdE_err)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    
if __name__ == '__main__':
    unittest.main()