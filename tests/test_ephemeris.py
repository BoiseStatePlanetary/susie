import sys
sys.path.append(".")
import unittest
import numpy as np
import matplotlib.pyplot as plt

from src.susie.timing_data import TimingData
from src.susie.ephemeris import Ephemeris, LinearModelEphemeris, QuadraticModelEphemeris, ModelEphemerisFactory
from scipy.optimize import curve_fit
test_epochs = np.array([0, 294, 298, 573])
test_mtts = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
test_mtts_err = np.array([0.00043, 0.00028, 0.00062, 0.00042])
test_tra_or_occ = np.array(['tra','occ','tra','occ'])
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
        """ Tests that ephemeris is an instance of LinearModelEphemeris
        """
        self.ephemeris = LinearModelEphemeris()
        self.assertIsInstance(self.ephemeris, LinearModelEphemeris)
    
    def test_linear_fit(self):
        """Tests that the lin_fit function works

            Creates a numpy.ndarray[float] with the length of the test data
        """
        linear_model = LinearModelEphemeris()
        # recalculate
        expected_result = np.array([-6.73466620e-05,  3.50213351e+02,  3.54978500e+02,  6.82559093e+02])
        result = linear_model.lin_fit(test_mtts, test_P_linear, test_T0_linear, test_tra_or_occ)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_lin_fit_model(self):
        """Testing that the dictionary parameters of fit model are equal to what they are suppose to be

            Tests the creation of a dictionary named return_data containing the linear fit model data in the order of:
            {'period': float,
            'period_err':float,
            'conjunction_time': float,
            'conjunction_time_err':float}    
        """
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
        self.assertEqual(popt[1], return_data['conjunction_time'])
        self.assertEqual(unc[1], return_data['conjunction_time_err'])


class TestQuadraticModelEphemeris(unittest.TestCase):

    def quad_fit_instantiation(self):
        """ Tests that ephemeris is an instance of QuadraticModelEphemeris
        """
        self.ephemeris = QuadraticModelEphemeris()
        self.assertIsInstance(self.ephemeris, QuadraticModelEphemeris)

    def test_quad_fit(self):
        """ Tests that the quad_fit function works

            Creates a numpy.ndarray[float] with the length of the test data
        """
        quadratic_model = QuadraticModelEphemeris()
        expected_result = np.array([-1.41514356e-06,  3.50213304e+02,  3.54978455e+02,  6.82559205e+02])
        result = quadratic_model.quad_fit(test_mtts,test_dPdE, test_P_quad, test_T0_quad)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_quad_fit_model(self):
        """Testing that the dictionary parameters of fit model are equal to what they are suppose to be

            Tests the creation of a dictionary named return_data containing the quadratic fit model data in the order of:
            {  'conjunction_time': float,
            'conjunction_time_err': float,
            'period': float,
            'period_err': float,
            'period_change_by_epoch': float,
            'period_change_by_epoch_err': float,
        """
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

# Do I need to create a test for this class????
# class TestModelEphemerisFactory(unittest.TestCase):
    def model_no_errors(self):
        models = {
            'linear': LinearModelEphemeris(),
            'quadratic': QuadraticModelEphemeris()
        }
        test_model_type = 'linear'
        self.assertTrue(test_model_type in models)
    
    def model_errors(self):
        models = {
            'linear': LinearModelEphemeris(),
            'quadratic': QuadraticModelEphemeris()
        }
        test_model_type = 'invaild_model'  
        with self.assertRaises(ValueError, msg = f"Invalid model type: {test_model_type}"):
            model = models[test_model_type]

class TestEphemeris(unittest.TestCase):

    def assertDictAlmostEqual(self, d1, d2, msg=None, places=7):
        """ Function used to check if the dictionarys are equal to eachother
        """
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
                # print(d1[key], d2[key])
                self.assertTrue(np.allclose(d1[key], d2[key], rtol=1e-05, atol=1e-08))
            else:
                self.assertAlmostEqual(d1[key], d2[key], places=places, msg=msg)
                

    def setUp(self):
       """ Sets up the intantiation of TransitTimes object and Ephemeris object

           Runs before every test in the TestEphemeris class
       """
       self.transit = TimingData('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
       self.assertIsInstance(self.transit, TimingData)
       self.ephemeris = Ephemeris(self.transit)
      

    def test_us_transit_times_instantiation(self):
        """ Unsuccessful instantiation of the transit times object within the Ephemeris class

            Need a TransitTimes object to run Ephemeris
        """
        with self.assertRaises(ValueError, msg = "Variable 'transit_times' expected type of object 'TransitTimes'."):
            self.ephemeris = Ephemeris(None)

    
    def test_get_model_parameters_linear(self):
        """ Tests the creation of the linear model parameters

            With the input of a linear model type, the linear model parameters dictionary is created
            The dictionary is the same one from fit_model in the LinearModelEphemeris
        """
        test_model_type= 'linear'
        model_parameters = self.ephemeris._get_model_parameters(test_model_type)
        expected_result = {
            'period': 1.0914223408652188,  
            'period_err': 9.998517417992763e-07,
            'conjunction_time': -6.734666196939187e-05, 
            'conjunction_time_err': 0.0003502975050463415
        }
        self.assertDictEqual(model_parameters,expected_result)   

    def test_get_model_parameters_quad(self):
        """ Tests the creation of the quadratic model parameters

            With the input of a quadratic model type, the quadratic model parameters dictionary is created
            The dictionary is the same one from fit_model in the QuadraticModelEphemeris
        """
        test_model_type = 'quadratic'
        model_parameters = self.ephemeris._get_model_parameters(test_model_type)   
        expected_result = {
            'conjunction_time': -1.415143555084551e-06,
            'conjunction_time_err': 0.00042940561938685084,
            'period': 1.0914215464474404,
            'period_err': 9.150815726215122e-06,
            'period_change_by_epoch': 2.7744598987630543e-09,
            'period_change_by_epoch_err': 3.188345582283935e-08,
        }
        self.assertDictEqual(model_parameters,expected_result)


    def test_k_value_linear(self):
        """ Tests the correct k value is returned given the linear model type

            The k value for a linear model is 2
        """
        test_model_type = 'linear'
        expected_result = 2
        result = self.ephemeris._get_k_value(test_model_type)
        self.assertEqual(result,expected_result)
    
    def test_k_value_quad(self):
        """ Tests the correct k value is returned given the quadratic model type

            The k value for a quadratic model is 3
        """
        test_model_type = 'quadratic'
        expected_result = 3
        result = self.ephemeris._get_k_value(test_model_type)
        self.assertEqual(result,expected_result)

    
    def test_calc_linear_model_uncertainties(self):
        """ Tests that the correct array of linear uncertainties are produced

            Produces a numpy array with the length of the epochs
        """
        expected_result = np.array([0.0003503 , 0.00045729, 0.00045988, 0.00067152])
        result = self.ephemeris._calc_linear_model_uncertainties(test_T0_err_linear, test_P_err_linear)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_quad_model_uncertainties(self):
        """ Tests that the correct array of quadratic uncertainties are produced

            Produces a numpy array with the length of the epochs
        """
        expected_result = np.array([0.00042941, 0.00305304, 0.00310238, 0.00742118])
        result = self.ephemeris._calc_quadratic_model_uncertainties(test_T0_err_quad, test_P_err_quad,test_dPdE_err)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_linear_ephemeris(self):
        """ Tests that the correct linear model data is produced

            The model data is a numpy array of calcuated mid transit times
        """
        expected_result = np.array([-6.73466620e-05, 3.20878101e+02, 3.25243790e+02, 6.25384934e+02])#test model data linear
        result = self.ephemeris._calc_linear_ephemeris(test_epochs, test_P_linear, test_T0_linear)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))


    def test_calc_quadratic_ephemeris(self):
        """ Tests that the correct quadratic model data is produced

            The model data is a numpy array of calcuated mid transit times
        """
        expected_result = np.array([-1.41514356e-06,  3.20878053e+02,  3.25243743e+02,  6.25385000e+02])#test model data quad
        result = self.ephemeris._calc_quadratic_ephemeris(test_epochs,test_P_quad,test_T0_quad,test_dPdE)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))


    def test_calc_chi_squared_linear(self):
        """ Tests the calculated chi squared value

            The linear chi squared value is a float that is calculated with the model data produced by test_calc_linear_ephemeris 
        """
        test_linear_model_data = np.array([-6.73466620e-05, 3.20878101e+02, 3.25243790e+02, 6.25384934e+02])
        expected_result = 0.29406284565290114
        result = self.ephemeris._calc_chi_squared(test_linear_model_data)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   


    def test_calc_chi_squared_quad(self):
        """ Tests the calculated chi squared value

            The quadratic chi squared value is a float that is calculated with the model data produced by test_calc_quadratic_ephemeris 
        """
        test_quad_model_data = np.array([-1.41514356e-06,  3.20878053e+02,  3.25243743e+02,  6.25385000e+02])
        expected_result = 0.20766342879185204
        result = self.ephemeris._calc_chi_squared(test_quad_model_data)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   
    
    def test_get_model_ephemeris_linear(self):
        """ Tests that the linear model type produces the linear model parameters with the linear model type and linear model data included

            Uses the test_get_model_parameters_linear and test_calc_linear_ephemeris to produce a dictionary with:
            {
            'period': float,  
            'period_err': float,
            'conjunction_time': float,
            'conjunction_time_err':  float,
            'model_type': 'linear', 
            'model_data': np.array
        }
        """
        test_model_type = 'linear'
        model_parameters_linear = {
            'period': 1.0914223408652188,  
            'period_err': 9.998517417992763e-07,
            'conjunction_time': -6.734666196939187e-05, 
            'conjunction_time_err':  0.0003502975050463415,
            'model_type': 'linear', 
            'model_data': ([-6.73466620e-05,  3.20878101e+02,  3.25243790e+02,  6.25384934e+02])
        }
        result = self.ephemeris.get_model_ephemeris(test_model_type)
        self.assertDictAlmostEqual(result, model_parameters_linear)

    def test_get_model_ephemeris_quad(self):
        """ Tests that the quadratic model type produces the quadratic model parameters with the quadratic model type and quadratic model data included

            Uses the test_get_model_parameters_linear and test_calc_linear_ephemeris to produce a dictionary with:
            {
            'period': float,  
            'period_err': float,
            'conjunction_time': float,
            'conjunction_time_err':  float,
            'period_change_by_epoch': float,
            'period_change_by_epoch_err': float,
            'model_type': 'quadratic', 
            'model_data': np.array
        }
        """
        test_model_type = 'quadratic'
        model_parameters_quad = {
            'conjunction_time': -1.415143555084551e-06,
            'conjunction_time_err': 0.00042940561938685084,
            'period': 1.0914215464474404,
            'period_err': 9.150815726215122e-06,
            'period_change_by_epoch': 2.7744598987630543e-09,
            'period_change_by_epoch_err': 3.188345582283935e-08,
            'model_type': 'quadratic',
            'model_data': ([-1.41514356e-06,  3.20878053e+02,  3.25243743e+02,  6.25385000e+02])
        }
        result = self.ephemeris.get_model_ephemeris(test_model_type)
        self.assertDictAlmostEqual(result, model_parameters_quad)

    def test_get_ephemeris_uncertainites_model_type_err(self):
        """ Unsuccessful test to calculate uncertainties

            Model type is needed
        """
        model_parameters_linear = {
            'period': 1.0914223408652188,  
            'period_err': 9.998517417992763e-07,
            'conjunction_time': -6.734666196939187e-05, 
            'conjunction_time_err':  0.0003502975050463415,
            'model_data': ([-6.73466620e-05,  3.20878101e+02,  3.25243790e+02,  6.25384934e+02])
        }
        with self.assertRaises(KeyError, msg = "Cannot find model type in model data. Please run the get_model_ephemeris method to return ephemeris fit parameters."):
            self.ephemeris.get_ephemeris_uncertainties(model_parameters_linear)
    
    def test_get_ephemeris_uncertainties_lin_err(self):
        """ Unsuccessful test to calculate uncertainties

            Period error and conjunction time error values are needed
        """
        model_parameters_linear = {
            'period': 1.0914223408652188,  
            'conjunction_time': -6.734666196939187e-05, 
            'model_type': 'linear', 
            'model_data': ([-6.73466620e-05,  3.20878101e+02,  3.25243790e+02,  6.25384934e+02])
        }
        with self.assertRaises(KeyError, msg = "Cannot find conjunction time and period errors in model data. Please run the get_model_ephemeris method with 'linear' model_type to return ephemeris fit parameters."):
            self.ephemeris.get_ephemeris_uncertainties(model_parameters_linear)


    def test_get_ephemeris_uncertainties_quad_err(self):
        """ Unsuccessful test to calculate uncertainties

            Conjunction time error, period error and period change by epoch error is needed
        """
        model_parameters_quad = {
            'conjunction_time': -1.415143555084551e-06,
            'period': 1.0914215464474404,
            'period_change_by_epoch': 2.7744598987630543e-09,
            'model_type': 'quadratic',
            'model_data': ([-1.41514356e-06,  3.20878053e+02,  3.25243743e+02,  6.25385000e+02])
        }
        with self.assertRaises(KeyError, msg = "Cannot find conjunction time, period, and/or period change by epoch errors in model data. Please run the get_model_ephemeris method with 'quadratic' model_type to return ephemeris fit parameters."):
            self.ephemeris.get_ephemeris_uncertainties(model_parameters_quad)

    
   
    def test_get_ephemeris_uncertainites_linear(self):
        """ Sucessful test to calculate linear uncertainties

            Expected result is the numpy array produced by test_calc_linear_model_uncertaintie
        """
        model_parameters_linear = {
            'period': 1.0914223408652188,  
            'period_err': 9.998517417992763e-07,
            'conjunction_time': -6.734666196939187e-05, 
            'conjunction_time_err': 0.0003502975050463415,
            'model_type': 'linear', 
            'model_data': ([-6.73466620e-05,  3.20878101e+02,  3.25243790e+02,  6.25384934e+02])
        }
        expected_result = np.array([0.0003503 , 0.00045729, 0.00045988, 0.00067152])
        self.ephemeris.get_ephemeris_uncertainties(model_parameters_linear)
        results = self.ephemeris.get_ephemeris_uncertainties(model_parameters_linear)
        self.assertTrue(np.allclose(expected_result, results, rtol=1e-05, atol=1e-08)) 
         
    def test_get_ephemeris_uncertainites_quad(self):
        """ Sucessful test to calculate quadratic uncertainties

            Expected result is the numpy array produced by test_calc_quadratic_model_uncertaintie
        """
        model_parameters_quad = {
            'conjunction_time': -1.415143555084551e-06,
            'conjunction_time_err': 0.00042940561938685084,
            'period': 1.0914215464474404,
            'period_err': 9.150815726215122e-06,
            'period_change_by_epoch': 2.7744598987630543e-09,
            'period_change_by_epoch_err': 3.188345582283935e-08,
            'model_type': 'quadratic',
            'model_data': ([-1.41514356e-06,  3.20878053e+02,  3.25243743e+02,  6.25385000e+02])
        }
        expected_result = np.array([0.00042941, 0.00305304, 0.00310238, 0.00742118])
        self.ephemeris.get_ephemeris_uncertainties(model_parameters_quad)
        results = self.ephemeris.get_ephemeris_uncertainties(model_parameters_quad)
        self.assertTrue(np.allclose(expected_result, results, rtol=1e-05, atol=1e-08)) 
         
    def test_calc_bic_lin(self):
        """ Tests the calculation of the linear bic

            Uses the linear k value and linear chi squared value
        """
        model_parameters_linear = {
            'period': 1.0914223408652188,  
            'period_err': 9.998517417992763e-07,
            'conjunction_time': -6.734666196939187e-05, 
            'conjunction_time_err': 0.0003502975050463415,
            'model_type': 'linear', 
            'model_data': ([-6.73466620e-05,  3.20878101e+02,  3.25243790e+02,  6.25384934e+02])
        }
        # k_value = 2
        # linear_chi_squared = 0.29406284565290114
        expected_result = 3.0666515678926825
        result = self.ephemeris.calc_bic(model_parameters_linear)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   
        
    def test_calc_bic_quad(self):
        """ Tests the calculation of the quadratic bic

            Uses the quadratic k value and quadratic chi squared value
        """
        model_parameters_quad = {
            'conjunction_time': -1.415143555084551e-06,
            'conjunction_time_err': 0.00042940561938685084,
            'period': 1.0914215464474404,
            'period_err': 9.150815726215122e-06,
            'period_change_by_epoch': 2.7744598987630543e-09,
            'period_change_by_epoch_err': 3.188345582283935e-08,
            'model_type': 'quadratic',
            'model_data': ([-1.41514356e-06,  3.20878053e+02,  3.25243743e+02,  6.25385000e+02])
        }
        # k_value = 3
        # quad_chi_squared = 0.20766342879185204
        expected_result = 4.3665465121515235
        result = self.ephemeris.calc_bic(model_parameters_quad)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08)) 
    
    def test_calc_delta_bic(self):
        """ Tests the calulation of the delta bic

            Uses both the quadratic bic and linear bic
        """
        # linear_bic = 3.0666515678926825
        # quad_bic = 4.3665465121515235
        expected_result = -1.299894944258841
        result = self.ephemeris.calc_delta_bic() 
        self.assertTrue(expected_result, result)

    

    if __name__ == '__main__':
            unittest.main()