import sys
sys.path.append("..")
import unittest
import numpy as np
from src.susie.timing_data import TimingData
from astroplan import Observer
from astropy.coordinates import EarthLocation
import astropy.units as u
from src.susie.ephemeris import Ephemeris, LinearModelEphemeris, QuadraticModelEphemeris, PrecessionModelEphemeris, ModelEphemerisFactory
np.random.seed(42)

# Test Data
epochs = np.array([-1640, -1018, -1011, 0, 13, 21, 2017, 2018])
mid_times = np.array([2454515.52496, 2455194.9344, 2455202.57625, 2456305.45536, 2456319.64424, 2456328.37556, 2458506.84758, 2458508.48459])
mid_time_errs = np.array([0.00043, 0.001, 0.0022, 0.00024, 0.00038, 0.00027, 0.00044, 0.00091])
tra_or_occs = np.array(['tra', 'occ', 'occ', 'tra', 'tra', 'tra', 'tra', 'occ'])
tra_or_occ_enum = np.array([0, 1, 1, 0, 0, 0, 0, 1])

# Declaring main base test case object so assertDictAlmostEqual is univerally accessible
class BaseTestCase(unittest.TestCase):
    def assertDictAlmostEqual(self, d1, d2, msg=None, places=6):
        # Helper function used to check if the dictionaries are equal to eachother
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
                self.assertTrue(np.allclose(d1[key], d2[key], rtol=1e-05, atol=1e-08))
            else:
                self.assertAlmostEqual(d1[key], d2[key], places=places, msg=msg)


# Testing the linear model object
class TestLinearModelEphemeris(BaseTestCase):
    def setUp(self):
        # Creates the linear model ephemeris object
        self.ephemeris = LinearModelEphemeris()
        self.epochs = epochs
        self.mid_times = mid_times
        self.mid_time_errs = mid_time_errs
        self.tra_or_occ = tra_or_occs
        self.tra_or_occ_enum = tra_or_occ_enum
        self.T0 = 0.0
        self.P = 1.0

    def test_lin_model_instantiation(self):
        # Tests that ephemeris is an instance of LinearModelEphemeris
        self.assertIsInstance(self.ephemeris, LinearModelEphemeris)
   
    def test_lin_fit(self):
        """Tests that the lin_fit function works.
           Creates a numpy.ndarray[float] with the length of the test data.
        """
        # The expected mid-times returned by the lin-fit equations
        expected_lin_fit = np.array([-1640, -1017.5, -1010.5, 0, 13, 21, 2017, 2018.5])
        actual_lin_fit = self.ephemeris.lin_fit(self.epochs, self.T0, self.P, self.tra_or_occ_enum)
        self.assertTrue(np.allclose(expected_lin_fit, actual_lin_fit, rtol=1e-05, atol=1e-08))

    def test_lin_fit_model(self):
        """Testing that the dictionary parameters of fit model are equal to what they are suppose to be

            Tests the creation of a dictionary named return_data containing the linear fit model data in the order of:
            {
             'period': float,
             'period_err': float,
             'conjunction_time': float,
             'conjunction_time_err': float
            }
        """
        lin_model_fit_result = self.ephemeris.fit_model(self.epochs, self.mid_times, self.mid_time_errs, self.tra_or_occ)
        lin_model_fit_return_data = {
            'period': 1.0914197876324176,
            'period_err': 3.8445359425767954e-07,
            'conjunction_time': 2456305.4551013056,
            'conjunction_time_err': 0.00034780710703679864
            }
        self.assertDictAlmostEqual(lin_model_fit_result, lin_model_fit_return_data)


# Testing the quadratic model object
class TestQuadraticModelEphemeris(BaseTestCase):
    def setUp(self):
        self.ephemeris = QuadraticModelEphemeris()
        self.epochs = epochs
        self.mid_times = mid_times
        self.mid_time_errs = mid_time_errs
        self.tra_or_occ = tra_or_occs
        self.tra_or_occ_enum = tra_or_occ_enum
        self.T0 = 0.0
        self.P = 1.0
        self.dPdE = 0.0

    def test_quad_fit_instantiation(self):
        # Tests that ephemeris is an instance of QuadraticModelEphemeris
        self.assertIsInstance(self.ephemeris, QuadraticModelEphemeris)

    def test_quad_fit(self):
        """Tests that the quad_fit function works.
           Creates a numpy.ndarray[float] with the length of the test data
        """
        expected_quad_fit = np.array([-1640, -1017.5, -1010.5, 0, 13, 21, 2017, 2018.5])
        actual_quad_fit = self.ephemeris.quad_fit(self.epochs, self.T0, self.P, self.dPdE, self.tra_or_occ_enum)
        self.assertTrue(np.allclose(expected_quad_fit, actual_quad_fit, rtol=1e-05, atol=1e-08))

    def test_quad_fit_model(self):
        """Testing that the dictionary parameters of fit model are equal to what they are suppose to be
            Tests the creation of a dictionary named return_data containing the quadratic fit model data in the order of:
            {  
            'conjunction_time': float,
            'conjunction_time_err': float,
            'period': float,
            'period_err': float,
            'period_change_by_epoch': float,
            'period_change_by_epoch_err': float,
            }
        """
        quad_model_fit_result = self.ephemeris.fit_model(self.epochs, self.mid_times, self.mid_time_errs, self.tra_or_occ)
        quad_model_return_data = {
            'period': 1.0914201416855518,
            'period_err': 1.0904667249408481e-07,
            'conjunction_time': 2456305.45556849,
            'conjunction_time_err': 0.00010574632139970142,
            'period_change_by_epoch': -1.194938064834552e-09,
            'period_change_by_epoch_err': 1.3310830044505155e-10
            }
        self.assertDictAlmostEqual(quad_model_fit_result, quad_model_return_data)


# Testing the Precession Model Object
class TestPrecessionModelEphemeris(BaseTestCase):
    def setUp(self):
        self.ephemeris = PrecessionModelEphemeris()
        self.epochs = epochs
        self.mid_times = mid_times
        self.mid_time_errs = mid_time_errs
        self.tra_or_occ = tra_or_occs
        self.tra_or_occ_enum = tra_or_occ_enum
        self.T0 = 0.0
        self.P = 1.0
        self.dPdE = 0.0
        self.e = 0.001
        self.w0 = 1.0
        self.dwdE = 0.001

    def test_precession_fit_instantiation(self):
        # Tests that ephemeris is an instance of PrecessionModelEphemeris
        self.assertIsInstance(self.ephemeris, PrecessionModelEphemeris)

    def test_anomalistic_period(self):
        expected_anomalistic_period = 1.00015918
        actual_anomalistic_period = self.ephemeris._anomalistic_period(self.P, self.dwdE)
        self.assertTrue(expected_anomalistic_period, actual_anomalistic_period)
    
    def test_pericenter(self):
        expected_result = np.array([-0.64, -0.018, -0.011, 1, 1.013, 1.021, 3.017, 3.018])
        result = self.ephemeris._pericenter(self.epochs, self.w0, self.dwdE)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))
        
    def test_precession_fit(self):
        """Tests that the precession_fit function works.
            Creates a numpy.ndarray[int] with the length of the test data
        """
        expected_result = np.array([-1640.000255, -1017.499602, -1010.499602, -1.72010942e-4, 12.99983149, 20.99983365, 2017.000316, 2018.499764])
        result = self.ephemeris.precession_fit(self.epochs, self.T0, self.P, self.e, self.w0, self.dwdE, self.tra_or_occ_enum)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_precession_fit_model(self):
        """Testing that the dictionary parameters of fit model are equal to what they are suppose to be
            Tests the creation of a dictionary named return_data containing the precession fit model data in the order of:
            {  
            'conjunction_time': float,
            'conjunction_time_err': float,
            'period': float,
            'period_err': float,
            'pericenter_change_by_epoch': float,
            'pericenter_change_by_epoch_err': float,
            'eccentricity': float,
            'eccentricity_err': float,
            'pericenter': float,
            'pericenter_err': float
            }
        """
        prec_model_result = self.ephemeris.fit_model(self.epochs, self.mid_times, self.mid_time_errs, self.tra_or_occ)
        prec_model_return_data = {
            'period': 1.091433725403284,
            'period_err': 8.897834090343257e-05,
            'conjunction_time': 2456305.48022954,
            'conjunction_time_err': 0.17582298200267588,
            'eccentricity': 0.9722406704173365,
            'eccentricity_err': 48715.865904328704,
            'pericenter': 55488688.58437893,
            'pericenter_err': 1431939.3571539088,
            'pericenter_change_by_epoch': -24013.745815457845,
            'pericenter_change_by_epoch_err': 776.7281997033724
            }
        self.assertDictAlmostEqual(prec_model_result, prec_model_return_data)


class TestModelEphemerisFactory(BaseTestCase):
    def setUp(self):
        # Initialize the Model Ephemeris Factory
        self.ephemeris = ModelEphemerisFactory()
        self.assertIsInstance(self.ephemeris, ModelEphemerisFactory)
        self.epochs = epochs
        self.mid_times = mid_times
        self.mid_time_errs = mid_time_errs
        self.tra_or_occ = tra_or_occs

    def test_model_no_errors_lin(self):
        # Check that a linear model dictionary is returned
        lin_model = self.ephemeris.create_model("linear", self.epochs, self.mid_times, self.mid_time_errs, self.tra_or_occ)
        return_data = {
            'period': 1.0914197876324176,
            'period_err': 3.8445359425767954e-07,
            'conjunction_time': 2456305.4551013056,
            'conjunction_time_err': 0.00034780710703679864
            }
        self.assertDictAlmostEqual(lin_model, return_data)

    def test_model_no_errors_quad(self):
        # Check that a quadratic model dictionary is returned
        quad_model = self.ephemeris.create_model('quadratic', self.epochs, self.mid_times, self.mid_time_errs, self.tra_or_occ)
        return_data = {
            'period': 1.0914201416855518,
            'period_err': 1.0904667249408481e-07,
            'conjunction_time': 2456305.45556849,
            'conjunction_time_err': 0.00010574632139970142,
            'period_change_by_epoch': -1.194938064834552e-09,
            'period_change_by_epoch_err': 1.3310830044505155e-10
            }
        self.assertDictAlmostEqual(quad_model, return_data)
    
    def test_model_no_errors_prec(self):
        # Check that a precession model dictionary is returned
        prec_model = self.ephemeris.create_model("precession", self.epochs, self.mid_times, self.mid_time_errs, self.tra_or_occ)
        return_data = {
            'period': 1.091433725403284,
            'period_err': 8.897834090343257e-05,
            'conjunction_time': 2456305.48022954,
            'conjunction_time_err': 0.17582298200267588,
            'eccentricity': 0.9722406704173365,
            'eccentricity_err': 48715.865904328704,
            'pericenter': 55488688.58437893,
            'pericenter_err': 1431939.3571539088,
            'pericenter_change_by_epoch': -24013.745815457845,
            'pericenter_change_by_epoch_err': 776.7281997033724
            }
        self.assertDictAlmostEqual(prec_model, return_data)

    def test_model_errors(self):
        # Check that ValueError is raised if not 'linear', 'quadratic', or 'precession'
        test_model_type = "invalid_model"
        with self.assertRaises(ValueError, msg=f"Invalid model type: {test_model_type}"):
            self.ephemeris.create_model(test_model_type, self.epochs, self.mid_times, self.mid_time_errs, self.tra_or_occ)


class TestEphemeris(BaseTestCase):            
    def setUp(self):
       """Sets up the intantiation of TimingData object and Ephemeris object.

           Runs before every test in the TestEphemeris class
       """
       self.timing_data = TimingData("jd", epochs, mid_times, mid_time_errs, tra_or_occs, time_scale="tdb")
       self.assertIsInstance(self.timing_data, TimingData)
       self.ephemeris = Ephemeris(self.timing_data)

    def test_us_transit_times_instantiation(self):
        """Unsuccessful instantiation of the timing data object within the Ephemeris class

            Need a TimingData object to run Ephemeris
        """
        with self.assertRaises(ValueError, msg="Variable 'timing_data' expected type of object 'TimingData'."):
            self.ephemeris = Ephemeris(None)

    def test_get_model_parameters_linear(self):
        """Tests the creation of the linear model parameters
            With the input of a linear model type, the linear model parameters dictionary is created
            The dictionary is the same one from fit_model in the LinearModelEphemeris
        """
        model_parameters = self.ephemeris._get_model_parameters("linear")
        expected_result = {
            'period': 1.0914197876324176,
            'period_err': 3.8445359425767954e-07,
            'conjunction_time': 2456305.4551013056,
            'conjunction_time_err': 0.00034780710703679864
            }
        self.assertDictAlmostEqual(model_parameters, expected_result)   

    def test_get_model_parameters_quad(self):
        """ Tests the creation of the quadratic model parameters

            With the input of a quadratic model type, the quadratic model parameters dictionary is created
            The dictionary is the same one from fit_model in the QuadraticModelEphemeris
        """
        model_parameters = self.ephemeris._get_model_parameters("quadratic")   
        expected_result = {
            'period': 1.0914201416855518,
            'period_err': 1.0904667249408481e-07,
            'conjunction_time': 2456305.45556849,
            'conjunction_time_err': 0.00010574632139970142,
            'period_change_by_epoch': -1.194938064834552e-09,
            'period_change_by_epoch_err': 1.3310830044505155e-10
            }
        self.assertDictAlmostEqual(model_parameters, expected_result)

    def test_get_model_parameters_prec(self):
        """ Tests the creation of the quadratic model parameters

            With the input of a quadratic model type, the quadratic model parameters dictionary is created
            The dictionary is the same one from fit_model in the QuadraticModelEphemeris
        """
        model_parameters = self.ephemeris._get_model_parameters("precession")   
        expected_result = {
            'period': 1.091433725403284,
            'period_err': 8.897834090343257e-05,
            'conjunction_time': 2456305.48022954,
            'conjunction_time_err': 0.17582298200267588,
            'eccentricity': 0.9722406704173365,
            'eccentricity_err': 48715.865904328704,
            'pericenter': 55488688.58437893,
            'pericenter_err': 1431939.3571539088,
            'pericenter_change_by_epoch': -24013.745815457845,
            'pericenter_change_by_epoch_err': 776.7281997033724
            }
        self.assertDictAlmostEqual(model_parameters, expected_result)

    def test_k_value_linear(self):
        """Tests the correct k value is returned given the linear model type
            The k value for linear model is 2
        """
        expected_result = 2
        result = self.ephemeris._get_k_value("linear")
        self.assertEqual(result, expected_result)
    
    def test_k_value_quad(self):
        """Tests the correct k value is returned given the quadratic model type
            The k value for a quadratic model is 3
        """
        expected_result = 3
        result = self.ephemeris._get_k_value("quadratic")
        self.assertEqual(result, expected_result)

    def test_k_value_prec(self):
        """Tests the correct k value is returned given the quadratic model type
            The k value for a precession model is 5
        """
        expected_result = 5
        result = self.ephemeris._get_k_value("precession")
        self.assertEqual(result, expected_result)

    def test_k_value_err(self):
        """Tests the correct k value is returned given the quadratic model type
            If not 'linear', 'quadratic', or 'precession', will return a ValueError
        """
        test_model_type = "invalid_type"
        with self.assertRaises(ValueError, msg="Only linear, quadratic, and precession models are supported at this time."):
            self.ephemeris._get_k_value(test_model_type)
    
    def test_calc_linear_model_uncertainties(self):
        """ Tests that the correct array of linear uncertainties are produced
            Produces a numpy array with the length of the epochs
            conjunction_time_err = 0.00034780710703679864
            period_err = 3.8445359425767954e-07
        """
        lin_model = self.ephemeris.fit_ephemeris("linear")
        expected_result = np.array([7.20072958e-4, 5.23443252e-4, 5.21435139e-4, 3.47807147e-4, 3.47843055e-4, 3.47900839e-4, 8.49871544e-4, 8.50397753e-4])
        result = self.ephemeris._calc_linear_model_uncertainties(lin_model["conjunction_time_err"], lin_model["period_err"])
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_quad_model_uncertainties(self):
        """ Tests that the correct array of quadratic uncertainties are produced
            Produces a numpy array with the length of the epochs
            conjunction_time_err = 0.00010574632139970142
            period_err = 1.0904667249408481e-07
            period_change_by_epoch_err = 1.3310830044505155e-10
        """
        quad_model = self.ephemeris.fit_ephemeris("quadratic")
        expected_result = np.array([2.74239386e-4, 1.68078507e-4, 1.67188748e-4, 1.05746321e-4, 1.05755824e-4, 1.05771118e-4, 3.64514509e-4, 3.64812697e-4])
        result = self.ephemeris._calc_quadratic_model_uncertainties(quad_model["conjunction_time_err"], quad_model["period_err"], quad_model["period_change_by_epoch_err"])
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_linear_ephemeris(self):
        """ Tests that the correct linear model data is produced
            The model data is a numpy array of calcuated mid transit times
            conjunction time = 2456305.4551013056
            period = 1.0914197876324176
        """
        lin_model = self.ephemeris.fit_ephemeris("linear")
        expected_result = np.array([2454515.527, 2455194.935, 2455202.575, 2456305.455, 2456319.644, 2456328.375, 2458506.849, 2458508.486])
        result = self.ephemeris._calc_linear_ephemeris(self.timing_data.epochs, lin_model["conjunction_time"], lin_model["period"])
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_quadratic_ephemeris(self):
        """ Tests that the correct quadratic model data is produced
            The model data is a numpy array of calcuated mid transit times
            conjunction_time = 2456305.45556849
            period = 1.0914201416855518
            period change by epoch = -1.194938064834552e-09
        """
        quad_model = self.ephemeris.fit_ephemeris("quadratic")
        expected_result = np.array([2454515.525, 2455194.935, 2455202.575, 2456305.456, 2456319.644, 2456328.375, 2458506.848, 2458508.485])
        result = self.ephemeris._calc_quadratic_ephemeris(self.timing_data.epochs, quad_model["conjunction_time"], quad_model["period"], quad_model["period_change_by_epoch"])
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_precession_ephemeris(self):
        """ Tests that the correct precession model data is produced
            The model data is a numpy array of calcuated mid transit times
            conjunction_time = 2456305.48022954
            period = 1.091433725403284
            eccentricity = 0.9722406704173365
            pericenter = 55488688.58437893
            pericenter change by epoch = -24013.745815457845
        """
        prec_model = self.ephemeris.fit_ephemeris("precession")
        expected_result = np.array([2454515.529, 2455194.401, 2455202.041, 2456305.48, 2456319.669, 2456328.4, 2458506.902, 2458507.994])
        result = self.ephemeris._calc_precession_ephemeris(self.timing_data.epochs, prec_model["conjunction_time"], prec_model["period"], prec_model["eccentricity"], prec_model["pericenter"], prec_model["pericenter_change_by_epoch"])
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_calc_chi_squared_linear(self):
        """ Tests the calculated chi squared value
            The linear chi squared value is a float that is calculated with the model data produced by test_calc_linear_ephemeris
            linear data = [2454515.52664959, 2455194.93546739, 2455202.5754059, 2456305.45510131, 
                           2456319.64355855, 2456328.37491685, 2458506.84881296, 2458508.48594264]
        """
        lin_model = self.ephemeris.fit_ephemeris("linear")
        expected_result = 36.83922943
        result = self.ephemeris._calc_chi_squared(lin_model["model_data"])
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   

    def test_calc_chi_squared_quad(self):
        """ Tests the calculated chi squared value
            The quadratic chi squared value is a float that is calculated with the model data produced by test_calc_quadratic_ephemeris 
            quad data = [2454515.52492917, 2455194.93495515, 2455202.57490463, 2456305.45556849,
                         2456319.64403023, 2456328.3753912, 2458506.84756359, 2458508.48469139]
        """
        quad_model = self.ephemeris.fit_ephemeris("quadratic")
        expected_result = 2.15135181
        result = self.ephemeris._calc_chi_squared(quad_model["model_data"])
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   
    
    def test_calc_chi_squared_prec(self):
        """ Tests the calculated chi squared value
            The precession chi squared value is a float that is calculated with the model data produced by test_calc_precession_ephemeris 
            prec data = [2454515.52900788, 2455194.40083254, 2455202.04080698, 2456305.48015054,
                         2456319.66889067, 2456328.40025227, 2458506.90196704, 2458507.99371183]
        """
        prec_model = self.ephemeris.fit_ephemeris("precession")
        expected_result = 673519.3095
        result = self.ephemeris._calc_chi_squared(prec_model["model_data"])
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

    def test_subract_linear_parameters(self):
        """ Tests subtracting linear terms from the model data
        data = [2454515.52492917, 2455194.93495515, 2455202.57490463, 2456305.45556849, 
                2456319.64403023, 2456328.3753912, 2458506.84756359, 2458508.48469139]
        conjunction time = 2456305.45556849
        period = 1.0914201416855518
        tra or occ = ['tra', 'occ', 'occ', 'tra', 'tra', 'tra', 'tra', 'occ']
        """
        quad_data = self.ephemeris.fit_ephemeris("quadratic")
        expected_result = np.array([-0.001606955695, -6.19174951e-4, -6.1068675e-4, 0, -1.01912175e-7, -2.6539659e-7, -0.002430679758, -0.002433092286])
        result = self.ephemeris._subtract_linear_parameters(quad_data["model_data"], quad_data["conjunction_time"], quad_data["period"], self.ephemeris.timing_data.epochs, self.ephemeris.timing_data.tra_or_occ)
        self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))
    
    def test_fit_ephemeris_linear(self):
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
        model_parameters_linear = {
            'period': 1.0914197876324176,
            'period_err': 3.8445359425767954e-07,
            'conjunction_time': 2456305.4551013056,
            'conjunction_time_err': 0.00034780710703679864,
            'model_type': 'linear',
            'model_data': [2454515.52664959, 2455194.93546739, 2455202.5754059 ,
                    2456305.45510131, 2456319.64355855, 2456328.37491685,
                    2458506.84881296, 2458508.48594264]}
        result = self.ephemeris.fit_ephemeris("linear")
        self.assertDictAlmostEqual(result, model_parameters_linear)

    def test_fit_ephemeris_quad(self):
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
        model_parameters_quad = {
            'period': 1.0914201416855518,
            'period_err': 1.0904667249408481e-07,
            'conjunction_time': 2456305.45556849,
            'conjunction_time_err': 0.00010574632139970142,
            'period_change_by_epoch': -1.194938064834552e-09,
            'period_change_by_epoch_err': 1.3310830044505155e-10,
            'model_type': 'quadratic',
            'model_data': [2454515.52492917, 2455194.93495515, 2455202.57490463,
                    2456305.45556849, 2456319.64403023, 2456328.3753912 ,
                    2458506.84756359, 2458508.48469139]}
        result = self.ephemeris.fit_ephemeris("quadratic")
        self.assertDictAlmostEqual(result, model_parameters_quad)

    def test_fit_ephemeris_prec(self):
        """ Tests that the precession model type produces the precession model parameters with the precession model type and precession model data included

            Produces a dictionary with:
            {
            'period': float,  
            'period_err': float,
            'conjunction_time': float,
            'conjunction_time_err':  float,
            'eccentricity': float, 
            'eccentricity_err': float,
            'pericenter': float,
            'pericenter_err': float,
            'pericenter_change_by_epoch': float,
            'pericenter_change_by_epoch_err': float,
            'model_type': 'precession', 
            'model_data': np.array
        }
        """
        model_parameters_prec = {
            'period': 1.091433725403284,
            'period_err': 8.897834090343257e-05,
            'conjunction_time': 2456305.48022954,
            'conjunction_time_err': 0.17582298200267588,
            'eccentricity': 0.9722406704173365,
            'eccentricity_err': 48715.865904328704,
            'pericenter': 55488688.58437893,
            'pericenter_err': 1431939.3571539088,
            'pericenter_change_by_epoch': -24013.745815457845,
            'pericenter_change_by_epoch_err': 776.7281997033724,
            'model_type': 'precession',
            'model_data': [2454515.52900788, 2455194.40083254, 2455202.04080698,
                    2456305.48015054, 2456319.66889067, 2456328.40025227,
                    2458506.90196704, 2458507.99371183]}
        result = self.ephemeris.fit_ephemeris("precession")
        self.assertDictAlmostEqual(result, model_parameters_prec)
    
    def test_get_ephemeris_uncertainites_model_type_err(self):
        """ Unsuccessful test to calculate uncertainties
            Model type is needed
        """
        # Just using random data
        model_parameters = {
            'period': 0.0,
            'period_err': 0.0,
            'conjunction_time': 0.0,
            'conjunction_time_err': 0.0,
            'model_data': np.array([0.0, 0.0])
        }
        with self.assertRaises(KeyError, msg="Cannot find model type in model data. Please run the fit_ephemeris method to return ephemeris fit parameters."):
            self.ephemeris.get_ephemeris_uncertainties(model_parameters)

    def test_get_ephemeris_uncertainties_lin_err(self):
        """ Unsuccessful test to calculate uncertainties
            Period error and conjunction time error values are needed
        """
        model_parameters = {
            'period': 1.0,
            'conjunction_time': 0.0,
            'model_type': 'linear',
            'model_data': np.array([0.0, 0.0])
        }
        with self.assertRaises(KeyError, msg="Cannot find conjunction time and period errors in model data. Please run the fit_ephemeris method with 'linear' model_type to return ephemeris fit parameters."):
            self.ephemeris.get_ephemeris_uncertainties(model_parameters)

    def test_get_ephemeris_uncertainties_quad_err(self):
        """ Unsuccessful test to calculate uncertainties
            Conjunction time error, period error and period change by epoch error is needed
        """
        model_parameters = {
        'period': 1.0,
        'conjunction_time': 0.0,
        'period_change_by_epoch': 0.0,
        'model_type': 'quadratic',
        'model_data': np.array([0.0, 0.0])
        }
        with self.assertRaises(KeyError, msg="Cannot find conjunction time, period, and/or period change by epoch errors in model data. Please run the fit_ephemeris method with 'quadratic' model_type to return ephemeris fit parameters."):
            self.ephemeris.get_ephemeris_uncertainties(model_parameters)

    def test_get_ephemeris_uncertainties_prec_err(self):
        """ Unsuccessful test to calculate uncertainties
            Conjunction time error, period error and period change by epoch error is needed
        """
        model_parameters = {
        'period': 1.0,
        'conjunction_time': 0.0,
        'period_change_by_epoch': 0.0,
        'model_type': 'precession',
        'model_data': np.array([0.0, 0.0])
        }
        with self.assertRaises(KeyError, msg="Cannot find conjunction time, period, eccentricity, pericenter, and/or pericenter change by epoch and/or their respective errors in model data. Please run the fit_ephemeris method with 'precession' model_type to return ephemeris fit parameters."):
            self.ephemeris.get_ephemeris_uncertainties(model_parameters)

    # def test_get_ephemeris_uncertainites_linear(self):
    #     """ Sucessful test to calculate linear uncertainties
    #         Expected result is the numpy array produced by test_calc_linear_model_uncertainties
    #     """
    #     model_parameters_linear = {
    #         'period': 1.0904734088754364,
    #         'period_err': 0.0006807481006299065,
    #         'conjunction_time': 2454515.423966982,
    #         'conjunction_time_err': 0.23692092991744518,
    #         'model_type': 'linear',
    #         'model_data': np.array([2454515.42396698, 2454836.5683859 , 2454840.38504283,
    #                 2455140.81046697])
    #     }
    #     expected_result = np.array([0.23692092, 0.31036088, 0.31190525, 0.45667354])
    #     self.ephemeris.get_ephemeris_uncertainties(model_parameters_linear)
    #     results = self.ephemeris.get_ephemeris_uncertainties(model_parameters_linear)
    #     self.assertTrue(np.allclose(expected_result, results, rtol=1e-05, atol=1e-08)) 
         
    # def test_get_ephemeris_uncertainites_quad(self):
    #     """ Sucessful test to calculate quadratic uncertainties

    #         Expected result is the numpy array produced by test_calc_quadratic_model_uncertaintie
    #     """
    #     model_parameters_quad = {
    #     'period': 1.0892663209112947,
    #     'period_err': 0.002368690041166098,
    #     'conjunction_time': 2454515.5241231285,
    #     'conjunction_time_err': 0.3467430587812461,
    #     'period_change_by_epoch': 4.223712653342504e-06,
    #     'period_change_by_epoch_err': 7.742732700893123e-06,
    #     'model_type': 'quadratic',
    #     'model_data': np.array([2454515.52412313, 2454836.49559505, 2454840.31302805,2455140.91174185])
    #     }
    #     expected_result = np.array([0.34674306,0.84783352,0.85829843,1.89241887])
    #     self.ephemeris.get_ephemeris_uncertainties(model_parameters_quad)
    #     results = self.ephemeris.get_ephemeris_uncertainties(model_parameters_quad)
    #     self.assertTrue(np.allclose(expected_result, results, rtol=1e-05, atol=1e-08)) 
         
    # def test_calc_bic_lin(self):
    #     """ Tests the calculation of the linear bic

    #         Uses the linear k value and linear chi squared value
    #     """
    #     model_parameters_linear = {
    #         'period': 1.0904734088754364,
    #         'period_err': 0.0006807481006299065,
    #         'conjunction_time': 2454515.423966982,
    #         'conjunction_time_err': 0.23692092991744518,
    #         'model_type': 'linear',
    #         'model_data': np.array([2454515.42396698, 2454836.5683859 , 2454840.38504283,
    #                 2455140.81046697])
    #     }
    #     # k_value = 2
    #     expected_result = 843769.0757319723
    #     result = self.ephemeris.calc_bic(model_parameters_linear)
    #     self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   
        
    # def test_calc_bic_quad(self):
    #     """ Tests the calculation of the quadratic bic

    #         Uses the quadratic k value and quadratic chi squared value
    #     """
    #     model_parameters_quad = {
    #     'period': 1.0892663209112947,
    #     'period_err': 0.002368690041166098,
    #     'conjunction_time': 2454515.5241231285,
    #     'conjunction_time_err': 0.3467430587812461,
    #     'period_change_by_epoch': 4.223712653342504e-06,
    #     'period_change_by_epoch_err': 7.742732700893123e-06,
    #     'model_type': 'quadratic',
    #     'model_data': np.array([2454515.52412313, 2454836.49559505, 2454840.31302805,2455140.91174185])
    #     }
    #     # k_value = 3
    #     expected_result = 650255.7398105409
    #     result = self.ephemeris.calc_bic(model_parameters_quad)
    #     self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08)) 
    

    # def test_calc_delta_bic(self):
    #     """ Tests the calulation of the delta bic

    #         Uses both the quadratic bic and linear bic
    #     """
    #     expected_result = 193513.33592143143
    #     result = self.ephemeris.calc_delta_bic(model_1, model_2)
    #     print(f"\n\n\n MEEP \n {expected_result}, {result} \n\n\n")
    #     self.assertTrue(expected_result, result)


# class TestPrecessionEphemeris(unittest.TestCase):        
#     def setUp(self):
#        """Sets up the intantiation of TimingData object and Ephemeris object.

#            Runs before every test in the TestEphemeris class
#        """
#        self.timing_data = TimingData('jd', test_epochs_precession, test_mtts_precession, test_mtts_err_precession, test_tra_or_occ_precession, time_scale='tdb')
#        self.assertIsInstance(self.timing_data, TimingData)
#        self.ephemeris = Ephemeris(self.timing_data)


#     def test_suc_t0_calc(self):
#         expected_result = 0.0000102238439463
#         result = self.ephemeris._calc_t0_model_uncertainty(test_T0_err_pre)
#         self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   
    
#     def test_suc_e_calc(self):
#         expected_result = 0.167733834964
#         result = self.ephemeris._calc_eccentricity_model_uncertainty(test_P_pre, test_dwdE, test_w, -1640, test_e_err)
#         self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))   
    
#     def test_suc_w_calc(self):
#         expected_result = 0.543545754247
#         result = self.ephemeris._calc_pericenter_model_uncertainty(test_e, test_P_pre, test_dwdE, test_w, -1640, test_w_err)
#         self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))     
    
#     def test_suc_dwdE_trans_calc(self):
#         expected_result = 0.00805534737998
#         result = self.ephemeris._calc_change_in_pericenter_transit_model_uncertainty(test_e, test_P_pre, test_dwdE, test_w, -1640, test_dwdE_err)
#         self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08)) 

#     def test_suc_dwdE_occ_calc(self):
#         expected_result = 0.00805539085732
#         result = self.ephemeris._calc_change_in_pericenter_transit_model_uncertainty(test_e, test_P_pre, test_dwdE, test_w, -1640, test_dwdE_err)
#         self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))

#     def test_suc_P_trans_calc(self):
#         expected_result = 0.000017849816617
#         result = self.ephemeris._calc_period_transit_model_uncertainty(test_e, test_dwdE, test_w, -1640, test_P_err_pre)
#         self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))  

#     def test_suc_P_occ_calc(self):
#         expected_result = 0.0000178498155019
#         result = self.ephemeris._calc_period_occ_model_uncertainty(test_e, test_dwdE, test_w, -1640, test_P_err_pre)
#         self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))    

#     def test_final_precession_uncertainty(self):
#         test_model_params = {'period': 1.091423398620008,
#             'period_err': 2.5761611591762386e-06,
#             'conjunction_time': 2456305.4591182857,
#             'conjunction_time_err': 0.0031974746201258552,
#             'eccentricity': 0.709249384779316,
#             'eccentricity_err': 21948.154831900636,
#             'pericenter': 201043977.85898417,
#             'pericenter_err': 594028.6944042204,
#             'pericenter_change_by_epoch': -116462.52579119639,
#             'pericenter_change_by_epoch_err': 44.094731428936605
#             }
#         expected_result = np.array([0.848152704559,0.500184477109,6.678399994, 7.03100035237, 7.43172845908, 0.517645163004, 7.30881685475])
#         result = self.ephemeris._calc_precession_model_uncertainties(test_model_params)
#         self.assertTrue(np.allclose(expected_result, result, rtol=1e-05, atol=1e-08))    


#     # Astroplan Test
#     # observer obj
#     def test_obs_obj_lat_long(self):
#         #  tests that if (lat, long) is passed in creates  observer obj
#         boise_state = (-116.2010,43.6023, 821)
#         timezone = 'US/Mountain'
#         observer = self.ephemeris._create_observer_obj(timezone, coords = boise_state)
#         self.assertEqual(observer.location.lat.deg, 43.6023)
#         self.assertEqual(observer.location.lon.deg, -116.2010)
    
#     def test_obs_obj_name(self):
#         # tests that if a name is passed in uses observer.at_site
#         name = 'Subaru'
#         timezone = 'US/Hawaii'
#         empty_coords = (None, None, None)
#         observer = self.ephemeris._create_observer_obj(timezone, coords=empty_coords, name=name)
#         self.assertEqual(observer.name, "Subaru")

#     def test_suc_return_obs_obj(self):
#         # tests for type astroplan.observer , name = str, coord = (float, float)
#         name = 'Subaru'
#         timezone = 'US/Hawaii'
#         empty_coords = (None, None, None)
#         observer = self.ephemeris._create_observer_obj(timezone, coords=empty_coords, name=name)
#         self.assertTrue(all(isinstance(value, str) for value in  observer.name))
        
#     def test_suc_return_obs_obj_float(self):
#         boise_state = (-116.2010,43.6023, 821.0)
#         timezone = 'US/Mountain'
#         observer = self.ephemeris._create_observer_obj(timezone, coords = boise_state)
#         self.assertTrue((isinstance(observer.location.lat.deg, (float, np.float64))))
#         self.assertTrue((isinstance(observer.location.lon.deg, (float, np.float64))))
#         # self.assertTrue((isinstance(observer.location.height (float, np.float64))))

#     def test_obs_obj_value_err(self):
#         # tests for value error if no name or lat or long
#         timezone = 'US/Hawaii'
#         empty_coords = (None, None, None)
#         with self.assertRaises(ValueError, msg="Observatory location must be specified with either (1) a site name specified by astropy.coordinates.EarthLocation.get_site_names() or (2) latitude and longitude in degrees as accepted by astropy.coordinates.Latitude and astropy.coordinates.Latitude."):
#             self.ephemeris._create_observer_obj(timezone, coords = empty_coords, name= None)

#     # target obj
#     def test_target_obj_ra_dec(self):
#         #  tests that if coords = (ra,dec) is passed in creates a fixed target with the ra and dec 
#         # check the ra and dec is returned
#         tres_3 = (268.0291,37.54633)
#         target = self.ephemeris._create_target_obj(coords = tres_3)
#         self.assertEqual(target.ra.deg, 268.0291)
#         self.assertEqual(target.dec.deg, 37.54633)
    
#     def test_target_obj_name(self):
#         # tests that if a name is passed in uses fixedTarget.from_name
#         tres_3 = (None,None)
#         target = self.ephemeris._create_target_obj(coords = tres_3, name='TrES-3b')
#         self.assertEqual(target.name, "TrES-3b")
    
#     def test_suc_return_target_obj(self):
#         # tests for type astroplan.FixedTarget , name = str, coord = (float, float)
#         name = 'TrES-3b'
#         empty_coords = (None, None)
#         target = self.ephemeris._create_target_obj(coords=empty_coords, name=name)
#         self.assertTrue(all(isinstance(value, str) for value in  target.name))
    
#     def test_suc_return_obj_float(self):
#         tres_3 = (268.0291,37.54633)
#         target = self.ephemeris._create_target_obj(coords = tres_3)
#         self.assertTrue((isinstance(target.ra.deg, (float, np.float64))))
#         self.assertTrue((isinstance(target.dec.deg, (float, np.float64))))
      
#     def test_target_obj_value_err(self):
#         # tests for value error if no name or ra or dec
#         empty_coords = (None, None, None)
#         with self.assertRaises(ValueError, msg="Object location must be specified with either (1) an valid object name or (2) right ascension and declination in degrees as accepted by astropy.coordinates.ra and astropy.coordinates.dec."):
#             self.ephemeris._create_target_obj(coords = empty_coords, name= None)
    
#     # query NASA
#     def test_input_value_err(self):
#         # tests for error if no coords or name
#         with self.assertRaises(ValueError, msg="Object must be specified with either (1) a recognized object name in the NASA Exoplanet Archive or (2) right ascension and declination in degrees as accepted by astropy.coordinates.ra and astropy.coordinates.dec."):
#             self.ephemeris._query_nasa_exoplanet_archive(None, ra=None, dec=None, select_query=None)
    
#     def test_nothing_found_name_value_err(self):
#         # tests for if nothing is found in the query for the name
#         with self.assertRaises(ValueError, msg=f"Nothing found for {'Earth'} in the NASA Exoplanet Archive. Please check that your object is accepted and contains data on the archive's homepage."):
#             self.ephemeris._query_nasa_exoplanet_archive("Earth", ra=None, dec=None, select_query=None)
   
#     def test_nothing_found_ra_dec_value_err(self):
#         # tests for if nothing is found in the query for the ra and dec
#         with self.assertRaises(ValueError, msg=f"Nothing found for the coordinates {0}, {0} in the NASA Exoplanet Archive. Please check that your values are correct and are in degrees as accepted by astropy.coordinates.ra and astropy.coordinates.dec."):
#             self.ephemeris._query_nasa_exoplanet_archive('ground', ra=0, dec=0, select_query=None)
   
#     def test_len_over_zero(self):
#         # tests the obj data length is not zero
#         pass

#     # eclipse duration
#     def test_eclipse_duration(self):
#         # tests calc eclipse duration
#         # trES_3 values
#         test_k = 0.1655
#         test_P = 1.30618581
#         test_a = 0.02282
#         test_b = 0.840
#         test_i = 81.85
#         test_R_star_a = 1/5.926
#         test_R_star = 1/(5.926 * (1/0.02282))
#         test_R_planet = 14.975
#         transit_duration = 3.296814064
#         result = self.ephemeris._calc_eclipse_duration(test_P, test_R_star, test_R_planet, test_a, test_b, test_i)
#         self.assertEqual(transit_duration, result)

#     # eclipse system params
#     def test_eclipse_params_type(self):
#         # tests type float 
#         pass
#     def test_eclipse_params_not_empty(self):
#         # tests that theres values but not using any values
#         pass
    
#     # observing schedule
#     def test_primary_eclipse(self):
#         # tests for the primary eclipse time using our data 
#         # is the last eclipse time
#         pass

#     def test_orbital_period(self):
#         # tests its a float, units and value
#         # from model period
#         pass


    if __name__ == '__main__':
            unittest.main()