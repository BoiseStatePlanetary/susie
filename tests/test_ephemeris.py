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

test_P_linear = 1.0914223408652188 # period linear
test_P_err_linear = 9.998517417992763e-07 # period error linear
test_T0_linear =  -6.734666196939187e-05 # conjunction time
test_T0_err_linear = 0.0003502975050463415 # conjunction time error

test_P_quad = 1.0914215464474404 #period quad
test_P_err_quad = 9.150815726215122e-06 # period err quad
test_dPdE = 2.7744598987630543e-09#period change by epoch
test_dPdE_err = 3.188345582283935e-08#period change by epoch error
test_T0_quad = -1.415143555084551e-06 #conjunction time quad
test_T0_err_quad = 0.00042940561938685084#conjunction time err quad

test_observed_data = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
test_uncertainties= np.array([0.00043, 0.00028, 0.00062, 0.00042])
       


class TestLinearModelEphemeris(unittest.TestCase):
    def linear_fit_instantiation(self):
        self.ephemeris = LinearModelEphemeris()
        self.assertIsInstance(self.emphemeris, LinearModelEphemeris)
    
    def test_linear_fit(self):
        linear_model = LinearModelEphemeris()
        expected_result =np.array([-6.73466620e-05,  3.50213351e+02,  3.54978500e+02,  6.82559093e+02])
        result = linear_model.lin_fit(test_mtts, test_P_linear, test_T0_linear)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_lin_fit_model(self):
        linear_model = LinearModelEphemeris()
        popt, pcov = curve_fit(linear_model.lin_fit, test_epochs, test_mtts, sigma=test_mtts_err, absolute_sigma=True)
        unc = np.sqrt(np.diag(pcov))
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
        expected_result= np.array([-1.41514356e-06,  3.50213304e+02,  3.54978455e+02,  6.82559205e+02])
        result = quadratic_model.quad_fit(test_mtts,test_dPdE, test_P_quad, test_T0_quad)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_quad_fit_model(self):
        quad_model = QuadraticModelEphemeris()
        popt, pcov = curve_fit(quad_model.quad_fit, test_epochs, test_mtts, sigma=test_mtts_err, absolute_sigma=True)
        unc = np.sqrt(np.diag(pcov))
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
    def assertDictAlmostEqual(self, d1, d2, msg=None, places=7):
        # check if both inputs are dicts
        self.assertIsInstance(d1, dict, 'First argument is not a dictionary')
        self.assertIsInstance(d2, dict, 'Second argument is not a dictionary')

        # check if both inputs have the same keys
        self.assertEqual(d1.keys(), d2.keys())

        # check each key
        for key, value in d1.items():
            if isinstance(value, dict):
                self.assertDictAlmostEqual(d1[key], d2[key], msg=msg)
            elif isinstance(value, np.ndarray):
                print(d1[key], d2[key])
                self.assertTrue(np.allclose(d1[key], d2[key], rtol=1e-05, atol=1e-08))
            else:
                self.assertAlmostEqual(d1[key], d2[key], places=places, msg=msg)
                

    def setUp(self):
       self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
       self.assertIsInstance(self.transit_times, TransitTimes)
       self.ephemeris=Ephemeris(self.transit_times)
      

    def transit_time_instantiation(self):
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertIsInstance(self.transit_times, TransitTimes)
        

    def test_us_transit_time_instantiation(self):
        with self.assertRaises(ValueError, msg="Variable 'transit_times' expected type of object 'TransitTimes'."):
            self.ephemeris = Ephemeris(None)

    
    def test_get_model_parameters_linear(self):
        test_model_type= 'linear'
        model_parameters = self.ephemeris._get_model_parameters(test_model_type)
        expected_result = {
            'period': [1.0914223408652188],  
            'period_err': [9.998517417992763e-07],
            'conjunction_time': [-6.734666196939187e-05], 
            'conjunction_time_err': [ 0.0003502975050463415]
        }
        self.assertDictEqual(model_parameters,expected_result)   

    def test_get_model_parameters_quad(self):
        test_model_type='quadratic'
        model_parameters=self.ephemeris._get_model_parameters(test_model_type)   
        expected_result={
            'conjunction_time': [-1.415143555084551e-06],
            'conjunction_time_err': [0.00042940561938685084],
            'period': [1.0914215464474404],
            'period_err': [9.150815726215122e-06],
            'period_change_by_epoch': [2.7744598987630543e-09],
            'period_change_by_epoch_err': [3.188345582283935e-08],
        }
        self.assertDictEqual(model_parameters,expected_result)


    def test_k_value_linear(self):
        test_model_type='linear'
        expected_result= 2
        result=self.ephemeris._get_k_value(test_model_type)
        self.assertEqual(result,expected_result)
    
    def test_k_value_quad(self):
        test_model_type='quadratic'
        expected_result= 3
        result=self.ephemeris._get_k_value(test_model_type)
        self.assertEqual(result,expected_result)

    
    def test_calc_linear_model_uncertainties(self):
        expected_result=np.array([0.0003503 , 0.00045729, 0.00045988, 0.00067152])
        result=self.ephemeris._calc_linear_model_uncertainties(test_T0_err_linear, test_P_err_linear)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_quad_model_uncertainties(self):
        expected_result=np.array([0.00042941, 0.00305304, 0.00310238, 0.00742118])
        result=self.ephemeris._calc_quadratic_model_uncertainties(test_T0_err_quad, test_P_err_quad,test_dPdE_err)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_linear_ephemeris(self):
        expected_result=np.array([-6.73466620e-05,  3.20878101e+02,  3.25243790e+02,  6.25384934e+02])#model data linear
        result=self.ephemeris._calc_linear_ephemeris(test_epochs, test_P_linear, test_T0_linear)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_quadratic_ephemeris(self):
        expected_result=np.array([-1.41514356e-06,  3.20878053e+02,  3.25243743e+02,  6.25385000e+02])#model data quad
        result=self.ephemeris._calc_quadratic_ephemeris(test_epochs,test_P_quad,test_T0_quad,test_dPdE)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

   #recalc
    def test_calc_chi_squared_linear(self):
        test_linear_model_data= np.array([-6.73466620e-05,  3.50213351e+02,  3.54978500e+02,  6.82559093e+02])
        expected_result= 155.20953457509313
        result=self.ephemeris._calc_chi_squared(test_linear_model_data)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   

    #recalc
    def test_calc_chi_squared_quad(self):
        test_quad_model_data= np.array([-1.41514356e-06,  3.20878053e+02,  3.25243743e+02,  6.25385000e+02])
        expected_result=53.74216679707304
        result=self.ephemeris._calc_chi_squared(test_quad_model_data)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   
    #recalc
    def test_get_model_ephemeris_linear(self):
        test_model_type = 'linear'
        model_parameters_linear = {
            'period': [1.0914223408652188],  
            'period_err': [9.998517417992763e-07],
            'conjunction_time': [-6.734666196939187e-05], 
            'conjunction_time_err': [ 0.0003502975050463415],
            'model_type': 'linear', 
            'model_data': ([-6.73466620e-05,  3.50213351e+02,  3.54978500e+02,  6.82559093e+02])
        }
        result = self.ephemeris.get_model_ephemeris(test_model_type)
        self.assertDictAlmostEqual(result, model_parameters_linear)

    def test_get_model_ephemeris_quad(self):
        test_model_type = 'quadratic'
        model_parameters_quad= {
            'conjunction_time': -1.415143555084551e-06,
            'conjunction_time_err': [0.00042940561938685084],
            'period': [1.0914215464474404],
            'period_err': [9.150815726215122e-06],
            'period_change_by_epoch': [2.7744598987630543e-09],
            'period_change_by_epoch_err': [3.188345582283935e-08],
            'model_type': 'quadratic',
            'model_data': ([-1.41514356e-06,  3.20878053e+02,  3.25243743e+02,  6.25385000e+02])
        }
        result = self.ephemeris.get_model_ephemeris(test_model_type)
        self.assertDictAlmostEqual(result, model_parameters_quad)

    def test_get_ephemeris_uncertainites_model_type_err(self):
        model_parameters_linear = {
            'period': [1.0914223408652188],  
            'period_err': [9.998517417992763e-07],
            'conjunction_time': [-6.734666196939187e-05], 
            'conjunction_time_err': [ 0.0003502975050463415],
            'model_data': ([-6.73466620e-05,  3.20878101e+02,  3.25243790e+02,  6.25384934e+02])
        }
        with self.assertRaises(KeyError, msg="Cannot find model type in model data. Please run the get_model_ephemeris method to return ephemeris fit parameters."):
            self.ephemeris.get_ephemeris_uncertainties(model_parameters_linear)
    
    def test_get_ephemeris_uncertainties_lin_err(self):
        model_parameters_linear = {
            'period': [1.0914223408652188],  
            'conjunction_time': [-6.734666196939187e-05], 
            'model_type': 'linear', 
            'model_data': ([-6.73466620e-05,  3.20878101e+02,  3.25243790e+02,  6.25384934e+02])
        }
        with self.assertRaises(KeyError, msg="Cannot find conjunction time and period errors in model data. Please run the get_model_ephemeris method with 'linear' model_type to return ephemeris fit parameters."):
            self.ephemeris.get_ephemeris_uncertainties(model_parameters_linear)


    def test_get_ephemeris_uncertainties_quad_err(self):
        model_parameters_quad= {
            'conjunction_time': -1.415143555084551e-06,
            'period': [1.0914215464474404],
            'period_change_by_epoch': [2.7744598987630543e-09],
            'model_type': 'quadratic',
            'model_data': ([-1.41514356e-06,  3.20878053e+02,  3.25243743e+02,  6.25385000e+02])
        }
        with self.assertRaises(KeyError, msg="Cannot find conjunction time, period, and/or period change by epoch errors in model data. Please run the get_model_ephemeris method with 'quadratic' model_type to return ephemeris fit parameters."):
            self.ephemeris.get_ephemeris_uncertainties(model_parameters_quad)

    ## recalc
    def test_get_ephemeris_uncertainites_linear(self):
         model_parameters_linear = {
            'period': [1.0914223408652188],  
            'period_err': [9.998517417992763e-07],
            'conjunction_time': [-6.734666196939187e-05], 
            'conjunction_time_err': [ 0.0003502975050463415],
            'model_type': 'linear', 
            'model_data': ([-6.73466620e-05,  3.20878101e+02,  3.25243790e+02,  6.25384934e+02])
        }
         expected_result = np.array([0.00146421, 0.00153663, 0.00153858, 0.00172645])
         results= self.ephemeris.get_ephemeris_uncertainties(model_parameters_linear)
         
    # def test_get_ephemeris_uncertainites_quad(self):
    #recalc
    def test_calc_bic_lin(self):
        model_parameters_linear = {
            'period': [1.0914223408652188],  
            'period_err': [9.998517417992763e-07],
            'conjunction_time': [-6.734666196939187e-05], 
            'conjunction_time_err': [ 0.0003502975050463415],
            'model_type': 'linear', 
            'model_data': ([-6.73466620e-05, 3.20880470e+02, 3.25246148e+02, 6.25386552e+02])##the first number should be 3.09284151e-03
        }
        k_value = 2
        chi_squared = 155.20953457509313
        expected_result = 157.98212329733292
        result= self.ephemeris.calc_bic(model_parameters_linear)
        print(result)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   
        # getting errors cuz the first term is different
    if __name__ == '__main__':
            unittest.main()