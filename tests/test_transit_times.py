import sys
sys.path.append(".")
from src.susie.timing_data import TimingData
import unittest
import numpy as np
from numpy.testing import assert_array_equal
import logging
from astropy import time
from astropy import coordinates
from astropy import units as u


# test_epochs = [0, 294, 298, 573, 579, 594, 602, 636, 637, 655, 677, 897, 901, 911, 912, 919, 941, 941, 963, 985, 992, 994, 995, 997, 1015, 1247, 1257, 1258, 1260, 1272, 1287, 1290, 1311, 1312, 1313, 1316, 1317, 1323, 1324, 1333, 1334, 1344, 1345, 1346, 1347, 1357, 1365, 1366, 1585, 1589, 1611, 1619, 1621, 1633, 1637, 1640, 1653, 1661, 1662, 1914, 1915, 1916, 1917, 1937, 1938, 1960, 1964, 1967, 1968, 1969, 1978, 1981, 1991, 1996, 2005, 2012, 2019, 2021, 2022, 2264, 2286, 2288, 2318, 2319, 2331, 2332, 2338, 2339, 2371, 2593, 2634, 2635, 2667, 2668, 2690, 2892, 2910, 2921, 2924, 2942, 2943, 2978, 2979, 2984, 2985, 2988, 2992, 2992, 2997, 2999, 3010, 3017, 3018, 3019, 3217, 3239, 3248, 3260, 3261, 3264, 3306, 3307, 3314, 3316, 3318, 3335, 3335, 3336, 3339, 3341, 3341, 3342, 3342, 3345, 3356, 3570, 3625, 3646, 3657]
# test_mtts = [0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421, 631.933999999892, 648.3059999998659, 657.0360000003129, 694.1440000003204, 695.2370000001974, 714.8820000002161, 738.8940000003204, 979.0049999998882, 983.3710000002757, 994.285000000149, 995.3769999998622, 1003.0160000002943, 1027.0270000002347, 1027.027999999933, 1051.0389999998733, 1075.0509999999776, 1082.691000000108, 1084.8730000001378, 1085.9650000003166, 1088.1480000000447, 1107.7930000000633, 1361.003000000026, 1371.9169999998994, 1373.0079999999143, 1375.191000000108, 1388.2889999998733, 1404.658999999985, 1407.933999999892, 1430.8530000001192, 1431.945000000298, 1433.036000000313, 1436.3100000000559, 1437.4020000002347, 1443.9500000001863, 1445.0419999998994, 1454.8640000000596, 1455.9560000002384, 1466.8700000001118, 1467.9620000002906, 1469.0530000003055, 1470.1450000000186, 1481.058999999892, 1489.7900000000373, 1490.8810000000522, 1729.9020000002347, 1734.2690000003204, 1758.2800000002608, 1767.0109999999404, 1769.194000000134, 1782.2910000002012, 1786.657000000123, 1789.9300000001676, 1804.1189999999478, 1812.851000000257, 1813.942000000272, 2088.9799999999814, 2090.0709999999963, 2091.163000000175, 2092.25400000019, 2114.0819999999367, 2115.1740000001155, 2139.185000000056, 2143.5509999999776, 2146.8250000001863, 2147.916000000201, 2149.0079999999143, 2158.8310000002384, 2162.1049999999814, 2173.0190000003204, 2178.4769999999553, 2188.2990000001155, 2195.939000000246, 2203.5789999999106, 2205.7620000001043, 2206.853000000119, 2470.9769999999553, 2494.9879999998957, 2497.1710000000894, 2529.913000000175, 2531.0049999998882, 2544.1019999999553, 2545.19299999997, 2551.7420000000857, 2552.8330000001006, 2587.7590000000782, 2830.0540000000037, 2874.8020000001416, 2875.8930000001565, 2910.81799999997, 2911.910000000149, 2935.9210000000894, 3156.388000000268, 3176.033999999985, 3188.0389999998733, 3191.313000000082, 3210.9590000002645, 3212.0500000002794, 3250.25, 3251.341000000015, 3256.7990000001155, 3257.8900000001304, 3261.1639999998733, 3265.529000000097, 3265.530999999959, 3270.9870000001974, 3273.1699999999255, 3285.1750000002794, 3292.814999999944, 3293.907000000123, 3294.998000000138, 3511.098999999929, 3535.1099999998696, 3544.933999999892, 3558.0300000002608, 3559.121999999974, 3562.3960000001825, 3608.2349999998696, 3609.3270000000484, 3616.966000000015, 3619.149999999907, 3621.3330000001006, 3639.885000000242, 3639.8870000001043, 3640.978000000119, 3644.253000000026, 3646.435000000056, 3646.435000000056, 3647.526000000071, 3647.526000000071, 3650.8009999999776, 3662.805999999866, 3896.3700000001118, 3956.3980000000447, 3979.31799999997, 3991.323000000324]
# test_mtts_err = [0.00043, 0.00028, 0.00062, 0.00042, 0.00043, 0.00032, 0.00036, 0.00046, 0.00041, 0.00019, 0.00043, 0.00072, 0.00079, 0.00037, 0.00031, 0.0004, 0.0004, 0.00028, 0.00028, 0.00068, 0.00035, 0.00029, 0.00024, 0.00029, 0.00039, 0.00027, 0.00021, 0.00027, 0.00024, 0.00032, 0.00031, 0.00022, 0.00018, 0.00017, 0.00033, 0.00011, 0.0001, 0.00017, 0.00032, 0.00039, 0.00035, 0.00034, 0.00035, 0.00032, 0.00042, 0.00037, 0.00037, 0.00031, 0.00033, 0.00039, 0.0003, 0.0003, 0.0003, 0.0003, 0.00046, 0.00024, 0.00038, 0.00027, 0.00029, 0.00021, 0.0003, 0.00033, 0.00071, 0.00019, 0.00043, 0.00034, 0.00034, 0.00019, 0.00019, 0.00031, 0.00028, 0.00032, 0.0004, 0.00029, 0.00029, 0.00025, 0.00034, 0.00034, 0.00046, 0.00043, 0.00039, 0.00049, 0.00046, 0.00049, 0.00035, 0.00036, 0.00022, 0.0002, 0.00031, 0.00042, 0.00033, 0.00033, 0.00055, 0.00023, 0.00021, 0.00035, 0.00025, 0.00034, 0.00037, 0.00028, 0.00023, 0.00028, 0.00039, 0.00024, 0.00022, 0.00029, 0.00043, 0.00036, 0.00026, 0.00048, 0.00032, 0.0004, 0.00018, 0.00021, 0.00056, 0.00023, 0.0003, 0.00022, 0.00034, 0.00028, 0.00027, 0.00035, 0.00031, 0.00032, 0.00033, 0.0005, 0.00031, 0.00032, 0.00091, 0.00035, 0.00026, 0.00021, 0.00034, 0.00034, 0.00038, 0.0004, 0.00026, 0.0003, 0.00044]
test_epochs = np.array([0, 294, 298, 573])
test_mtts = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
test_mtts_err = np.array([0.00043, 0.00028, 0.00062, 0.00042])
tra_or_occ = np.array(['tra','occ','tra','occ'])
timing_data = TimingData('jd', test_epochs, test_mtts, test_mtts_err, tra_or_occ,'utc',object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
expected_mid_times_obj = time.Time(test_mtts,format = 'jd',scale = 'utc')
expected_mid_times_err_obj = time.Time(test_mtts_err,format = 'jd',scale = 'utc')
        
class TestTimingData(unittest.TestCase):
       
    def test_successful_instantiation_jd_tdb_timescale(self):
        """Successful creation of the TimingData object with proper timescale

            Creating the TimingData object with the proper timescale of JD TDB which is
            barycentric Julian Date. Also including uncertainties.
        """
        # Should not get any errors, the epochs and transit times should be the same as they are inputted
        self.timing_data = TimingData('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertIsInstance(self.timing_data, TimingData)  # Check if the object is an instance of TransitTimes
        shifted_epochs = test_epochs - np.min(test_epochs) 
        self.assertTrue(np.array_equal(self.timing_data.epochs, shifted_epochs))  # Check if epochs remain unchanged
        self.assertTrue(np.array_equal(self.timing_data.mid_times, test_mtts))  # Check mid_transit_times
        self.assertTrue(np.array_equal(self.timing_data.mid_time_uncertainties, test_mtts_err))  # Check uncertainties

    def test_s_init_jd_tdb_no_uncertainties(self):
        """ Successful creation of the TimingData object with proper timescale

            Creating the TimingData object with the proper timescale of JD TDB which is
            barycentric Julian Date. Not including uncertainties.
        """
        # Should not get any errors, the epochs and transit times should be the same as they are inputted
        self.timing_data = TimingData('jd', test_epochs, test_mtts, time_scale='tdb')
        self.assertIsInstance(self.timing_data, TimingData )  # Check if the object is an instance of TransitTimes
        shifted_epochs = test_epochs - np.min(test_epochs)
        shifted_mtt = test_mtts - np.min(test_mtts)
        new_uncertainties = np.ones_like(test_epochs, dtype=float)
        self.assertTrue(np.array_equal(self.timing_data.epochs, shifted_epochs))  # Check if epochs remain unchanged
        self.assertTrue(np.array_equal(self.timing_data.mid_times, shifted_mtt))  # Check mid_transit_times
        self.assertTrue(np.array_equal(self.timing_data.mid_time_uncertainties, new_uncertainties))  # Check uncertainties chage this back!!!

    def test_suc_arrays(self):
        """ Successful test to check that epochs, mid times, mid time uncertainties, and tra_or_occ are all type np.arrays

        """
        self.timing_data =  TimingData('jd', test_epochs, test_mtts, test_mtts_err, tra_or_occ, time_scale='tdb')
        self.assertTrue(isinstance( self.timing_data.epochs, np.ndarray))
        self.assertTrue(isinstance(self.timing_data.mid_times, np.ndarray))
        self.assertTrue(isinstance(self.timing_data.mid_time_uncertainties, np.ndarray))
        self.assertTrue(isinstance(self.timing_data.tra_or_occ, np.ndarray))
   
    def test_us_epochs_arr_type_str(self):
        """ Unsuccessful test to check for numpy array validation for the epochs.
            
            The epochs are strings instead of numpy array and should raise an error.
        """
        string_test_epochs_arr = str(test_epochs)
        with self.assertRaises(TypeError, msg="The variable 'epochs' expected a NumPy array (np.ndarray) but received a different data type"):
             TimingData('jd', string_test_epochs_arr, test_mtts, test_mtts_err, time_scale='tdb')
     
    def test_us_mtts_arr_type_str(self):
        """ Unsuccessful test to check for numpy array validation for the mid times.

            The mid times are strings instead of numpy array and should raise an error.
        """
        string_test_mtts_arr = str(test_mtts)
        with self.assertRaises(TypeError, msg="The variable 'mid_transit_times' expected a NumPy array (np.ndarray) but received a different data type"):
             TimingData('jd', test_epochs, string_test_mtts_arr, test_mtts_err, time_scale='tdb')
    
    def test_us_mtts_err_arr_type_str(self):
        """ Unsuccessful test to check for numpy array validation for the mid time uncertainties.
        
            The mid time uncertainites are strings instead of numpy array and should raise an error.
        """
        string_test_mtts_err_arr = str(test_mtts_err)
        with self.assertRaises(TypeError, msg="The variable 'mid_transit_times_uncertainties' expected a NumPy array (np.ndarray) but received a different data type"):
             TimingData('jd', test_epochs, test_mtts, string_test_mtts_err_arr, time_scale='tdb')  
   
    def test_s_vars_value_types(self):
        """ Successful test to check the correct data type

            Epochs should be integers, mid times should be floats, mid time uncertainties should be floats and tra_or_occ should be strings.
        """
        self.timing_data =  TimingData('jd', test_epochs, test_mtts, test_mtts_err, tra_or_occ, time_scale='tdb')
        self.assertTrue(all(isinstance(value, (int, np.int64)) for value in self.timing_data.epochs))
        self.assertTrue(all(isinstance(value, float) for value in self.timing_data.mid_times))
        self.assertTrue(all(isinstance(value, float) for value in self.timing_data.mid_time_uncertainties))
        self.assertTrue(all(isinstance(value, str) for value in self.timing_data.tra_or_occ))
   
    def test_us_epochs_value_types_float(self):
        """ Unsuccessful test to check for data type validation of the epochs.

            The epochs are floats instead of integers and should raise an error.
        """
        float_test_epochs = test_epochs.astype(float)
        with self.assertRaises(TypeError, msg="All values in 'epochs' must be of type int."):
             TimingData('jd', float_test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
    
    def test_us_mtts_value_types_int(self):
        """ Unsuccessful test to check for data type validation of the mid times.

            The mid times are integers instead of floats and should raise an error.
        """
        int_test_mtts= test_mtts.astype(int)
        with self.assertRaises(TypeError, msg="All values in 'mid_transit_times' must be of type float."):
             TimingData('jd', test_epochs, int_test_mtts, test_mtts_err, time_scale='tdb')
    
    def test_us_mtts_err_value_types_int(self):
        """ Unsuccessful test to check for data type validation of the mid time uncertainties.

            The mid time uncertainties are integers instead of floats and should raise an error.
        """
        int_test_mtts_err= test_mtts_err.astype(int)
        with self.assertRaises(TypeError, msg="All values in 'mid_transit_times_uncertainties' must be of type float."):
             TimingData('jd', test_epochs, test_mtts, int_test_mtts_err, time_scale='tdb')

    def test_shifted_epochs_zero(self):
        """ Successful test to check the shifted epochs function works when the epochs start with 0.

            The epochs should remain the same as the array already starts at zero.
        """
        test_epochs_zero = np.array([0, 294, 298, 573]).astype(int)
        shifted_epochs_zero = np.array([0, 294, 298, 573]).astype(int)
        self.timing_data =  TimingData('jd', test_epochs_zero, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.timing_data.epochs, shifted_epochs_zero))
    
    def test_shifted_epochs_pos(self):
        """ Successful test to check the shifted epochs function works when the epochs start with a positive number.

            The epochs should shift to start with zero.
        """
        test_epochs_pos = np.array([1, 294, 298, 573]).astype(int)
        shifted_epochs_pos = np.array([0, 293, 297, 572]).astype(int)
        self.timing_data =  TimingData('jd', test_epochs_pos, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.timing_data.epochs, shifted_epochs_pos))

    def test_shifted_epochs_neg(self):
        """ Successful test to check the shifted epochs function works when the epochs start with a negative number.

            The epochs should shift to start with zero.
        """
        test_epochs_neg = np.array([-1, 294, 298, 573]).astype(int)
        shifted_epochs_neg = np.array([0, 295, 299, 574]).astype(int)
        self.timing_data =  TimingData('jd', test_epochs_neg, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.timing_data.epochs, shifted_epochs_neg))

    def test_shifted_mtts_zero(self):
        """ Successful test to check the shifted mid times function works when the mid times start with 0.

            The mid times should remain the same as the array already starts at zero.
        """
        test_mtts_zero = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
        shifted_mtts_zero = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
        self.timing_data =  TimingData('jd', test_epochs, test_mtts_zero, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.timing_data.mid_times, shifted_mtts_zero))

    def test_shifted_mtts_pos(self):
        """ Successful test to check the shifted mid times function works when the mid times start with a positive number.

            The mid times should shift to start with zero.
        """
        test_mtts_pos = np.array([1.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
        shifted_mtts_pos = np.array([0.0, 319.8780000000261, 324.24399999994785, 624.3850000002421])
        self.timing_data =  TimingData('jd', test_epochs, test_mtts_pos, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.timing_data.mid_times, shifted_mtts_pos))
    
    def test_shifted_mtts_neg(self):
        """ Successful test to check the shifted mid times function works when the mid times start with a negative number.

            The mid times should shift to start with zero.
        """
        test_mtts_neg = np.array([-1.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
        shifted_mtts_neg = np.array([0.0, 321.8780000000261, 326.24399999994785, 626.3850000002421])
        self.timing_data =  TimingData('jd', test_epochs, test_mtts_neg, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.timing_data.mid_times, shifted_mtts_neg))

#<————————————————————————————————————————————————————————————————————————————————————————>
    def test_no_mtts_err(self):
        """ Successful test for when no mid time uncertainties are given.

            If no mid time uncertainties are given the certainties will be an array ones with a data type of floats and the length of the mid times array.
        """
        test_mtts_err = None
        self.timing_data =  TimingData('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        if test_mtts_err is None:
            new_uncertainities= np.ones_like(test_epochs,dtype=float)
            self.assertTrue(np.all(new_uncertainities==np.ones_like(test_epochs,dtype=float)))
        
    def test_mid_transit_err_ones(self):
        """ Successful test for when the mid time uncertainties are an array of ones.
        
        """
        new_test_mtts_err=np.ones_like(test_mtts_err)
        self.timing_data= TimingData('jd', test_epochs, test_mtts, new_test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.timing_data.mid_time_uncertainties,new_test_mtts_err))

    def test_mid_transit_err_neg(self):
        """ Unsuccessful test for when the mid time uncertainties are negative.

            The mid time uncertainties must be postive and will raise an error if the values are negative.
        """
        test_mtts_err_neg= np.array([-0.00043, -0.00028, -0.00062, -0.00042])
        with self.assertRaises(ValueError, msg="The 'mid_transit_times_uncertainties' array must contain non-negative and non-zero values."):
             TimingData('jd', test_epochs, test_mtts, test_mtts_err_neg, time_scale='tdb')  
      
    def test_mid_transit_err_zero(self):
        """ Unsuccessful test for when the mid time uncertainties are zero.

            The mid time uncertianties must be postive and greater than zero and will raise an error if the values are zero.
        """
        test_mtts_err_zero= np.array([0.,0.,0.,0.])
        with self.assertRaises(ValueError, msg="The 'mid_transit_times_uncertainties' array must contain non-negative and non-zero values."):
            TimingData('jd', test_epochs, test_mtts, test_mtts_err_zero, time_scale='tdb')  

    def test_mid_transit_err_self(self):
        """ Successful test for postive and greater than zero mid time uncertainties.

        """
        self.timing_data =  TimingData('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.timing_data.mid_time_uncertainties, test_mtts_err))
   
    def test_variable_shape(self):
        """ Successful test to check that all of the varibles have the same shape.

        """
        self.timing_data =  TimingData('jd', test_epochs, test_mtts, test_mtts_err, tra_or_occ, time_scale='tdb')
        self.assertEqual(test_epochs.shape, test_mtts.shape)
        self.assertEqual(test_epochs.shape, test_mtts_err.shape)
        self.assertEqual(test_epochs.shape, tra_or_occ.shape)

    def test_variable_shape_fail(self):
        """ Unsuccessful test of the varibles shape.

            The epochs, mid times and mid time uncertainties all have different shapes.
        """
        new_test_epochs= np.array([0, 298, 573])  
        new_test_mtts= np.array([0.0, 625.3850000002421])
        new_tra_or_occ = np.array(['tra','tra','occ','occ','occ'])
        with self.assertRaises(ValueError, msg="Shapes of 'epochs', 'mid_transit_times', and 'mid_transit_times_uncertainties' arrays do not match."):
             TimingData('jd', new_test_epochs, new_test_mtts, test_mtts_err, new_tra_or_occ, time_scale='tdb')  
    
    def test_variable_tra_and_occ_shape_fail(self):
        """ Unsuccessful test of the varibles shape.

            The epochs, mid times and mid time uncertainties all have different shapes.
        """
        new_test_epochs= np.array([0, 298, 573])  
        new_test_mtts= np.array([0.0, 625.3850000002421])
        new_tra_or_occ = np.array(['tra','tra','occ','occ','occ'])
        with self.assertRaises(ValueError, msg="Shapes of 'tra_or_occ', 'mid_time_uncertainties', 'mid_times', and 'epochs' arrays do not match."):
             TimingData('jd', new_test_epochs, new_test_mtts, test_mtts_err, new_tra_or_occ, time_scale='tdb')  

    def successful_no_nan_values(self):
        """ Successful test for no Not a Number (NaN) values.

            No NaN values in epochs, mid times and mid times uncertainties.
        """
        self.timing_data =  TimingData('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertNotIn(np.nan,test_epochs)
        self.assertNotIn(np.nan,test_mtts)
        self.assertNotIn(np.nan,test_mtts_err)

    def test_mtts_nan(self):
        """ Unsuccessful test to check NaN values in mid times.

            Mid times cannot have any NaN values within the array.
        """
        new_test_mtts=np.array([0., np.nan , 298. ,573.], dtype=float)
        with self.assertRaises(ValueError, msg="The 'mid_transit_times' array contains NaN (Not-a-Number) values."):
             TimingData('jd', test_epochs, new_test_mtts, test_mtts_err, time_scale='tdb')  
    
    def test_mtts_err_nan(self):
        """ Unsuccessful test to check NaN values in mid times uncertainties.

            Mid times uncertainties cannot have any NaN values within the array.
        """
        new_test_mtts_err=np.array([0.00043, np.nan, 0.00062, 0.00042], dtype=float)
        with self.assertRaises(ValueError, msg="The 'mid_transit_times_uncertainties' array contains NaN (Not-a-Number) values."):
             TimingData('jd', test_epochs, test_mtts, new_test_mtts_err, time_scale='tdb')  
  
    def test_validate_times_obj_coords_err(self):
        """ Unsuccessful test to check the object coordinates within the Timing Data.

            The right ascension and declination of the object must be given.
        """
        with self.assertRaises(ValueError, msg="Recieved None for object right ascension and/or declination. Please enter ICRS coordinate values in degrees for object_ra and object_dec for TransitTime object."):
            TimingData('jd', test_epochs, test_mtts, test_mtts_err, time_scale='utc')

    def test_validate_times_obs_coords_logging_err(self):
        """ Test to that that the proper warnings come up for having no observatory coordinates.

            The logging warnings for the time format/time scale and the ICRS coordinates should pop up no matter what.
            The logging warning for the observatory coordinates only pops up when no coordinates are used. The observatory coordinates used becomes the graviational center of the Earth.
        """
        timing_data = TimingData('jd', test_epochs, test_mtts, test_mtts_err, tra_or_occ, 'utc', object_ra=97.64, object_dec=29.67, observatory_lat=None, observatory_lon=None)
        with self.assertLogs('lumberjack', level='WARNING') as cm:
            logger = logging.getLogger('lumberjack')
            logger.warning('Recieved time format jd and time scale utc. Correcting all times to BJD timing system with TDB time scale. If no time scale is given, default is automatically assigned to UTC. If this is incorrect, please set the time format and time scale for TransitTime object.')
            logger.warning("Unable to process observatory coordinates (None,None). "
                             "Using gravitational center of Earth.")
            logger.warning("Using ICRS coordinates in degrees of RA and Dec (97.64, 29.67) for time correction. "
                        "Using geodetic Earth coordinates in degrees of longitude and latitude (-116.21, 43.6) for time correction.")
        self.assertEqual(cm.output, ['WARNING:lumberjack:Recieved time format jd and time scale utc. Correcting all times to BJD timing system with TDB time scale. If no time scale is given, default is automatically assigned to UTC. If this is incorrect, please set the time format and time scale for TransitTime object.','WARNING:lumberjack:Unable to process observatory coordinates (None,None). Using gravitational center of Earth.','WARNING:lumberjack:Using ICRS coordinates in degrees of RA and Dec (97.64, 29.67) for time correction. Using geodetic Earth coordinates in degrees of longitude and latitude (-116.21, 43.6) for time correction.']) 
        
    def test_validate_times_logging_err(self):
        """ Test to that that the proper warnings come up for having observatory coordinates.

            The logging warnings for the time format/time scale and the ICRS coordinates should pop up no matter what.
        """
        timing_data = TimingData('jd', test_epochs, test_mtts, test_mtts_err, tra_or_occ, 'utc', object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
        with self.assertLogs('lumberjack', level='WARNING') as cm:
            logger = logging.getLogger('lumberjack')
            logger.warning('Recieved time format jd and time scale utc. Correcting all times to BJD timing system with TDB time scale. If no time scale is given, default is automatically assigned to UTC. If this is incorrect, please set the time format and time scale for TransitTime object.')
            logger.warning('Using ICRS coordinates in degrees of RA and Dec (97.64, 29.67) for time correction. Using geodetic Earth coordinates in degrees of longitude and latitude (-116.21, 43.6) for time correction.')
        self.assertEqual(cm.output, ['WARNING:lumberjack:Recieved time format jd and time scale utc. Correcting all times to BJD timing system with TDB time scale. If no time scale is given, default is automatically assigned to UTC. If this is incorrect, please set the time format and time scale for TransitTime object.','WARNING:lumberjack:Using ICRS coordinates in degrees of RA and Dec (97.64, 29.67) for time correction. Using geodetic Earth coordinates in degrees of longitude and latitude (-116.21, 43.6) for time correction.']) 
    
    def test_obs_geocentric_coords(self):
        """ Successful test to check that the correct observatory coordinates are produced by validate times.

            Testing to see that given a time scale/time format other that jd and tdb the validate times function produces the correct observatory coordinates.
            When given no coordinates the obsevatory coordinates produced should be the gravitational center of the Earth.
        """
        timing_data = TimingData('jd', test_epochs, test_mtts, test_mtts_err, tra_or_occ, 'utc', object_ra=97.64, object_dec=29.67, observatory_lat=None, observatory_lon=None)
        test_mid_times_obj = time.Time(test_mtts,format = 'jd',scale = 'utc')
        test_mid_times_err_obj = time.Time(test_mtts_err,format = 'jd',scale = 'utc')
        corrected_times = timing_data._validate_times(test_mid_times_obj, test_mid_times_err_obj,(97.64,29.67),(None,None))
        obs_location = (0., 0.)
        expected_obs_location = coordinates.EarthLocation.from_geocentric(0., 0., 0., unit=u.m)
        self.assertTrue(np.allclose(obs_location[0], expected_obs_location.x.value))
        self.assertTrue(np.allclose(obs_location[1], expected_obs_location.y.value))
       
    def test_obs_geodetic_coords(self):
        """ Successful test to check that the correct observatory coordinates are produced by validate times.

            Testing to see that given a time scale/time format other that jd and tdb the validate times function produces the correct observatory coordinates.
        """
        obs_coords = (-2042896.9,-4149886.2,4376818.8)
        expected_obs_location = coordinates.EarthLocation.from_geodetic(-116.21,43.61)
        self.assertTrue(np.allclose(obs_coords[0], expected_obs_location.x.value))
        self.assertTrue(np.allclose(obs_coords[1], expected_obs_location.y.value))
        self.assertTrue(np.allclose(obs_coords[2], expected_obs_location.z.value))

    def test_obj_coords(self):
        """ Successful test to check that the correct object coordinates are produced by validate times.

            Testing to see that given a time scale/time format other that jd and tdb the validate times function produces the correct object coordinates.
        """
        timing_data = TimingData('jd', test_epochs, test_mtts, test_mtts_err, tra_or_occ, 'utc', object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
        obj_coords = (97.64,29.67)
        expected_obj_location = coordinates.SkyCoord(ra=97.64, dec=29.67, unit='deg')
        self.assertEqual(obj_coords[0], expected_obj_location.ra.deg)
        self.assertEqual(obj_coords[1], expected_obj_location.dec.deg)

    def test_calc_bary_time_instantiation(self):
        """ Successful test for instantiation of timing data.

        """
        self.timing_data =  TimingData('jd', test_epochs, test_mtts, test_mtts_err, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
        self.assertIsInstance(self.timing_data,  TimingData)
    
    def test_calc_bary_time_uncertainties(self):
        """ Successful test to check correct process when the uncertainties are ones for the barycentric time function.

            If the uncertainties are ones, there is no correction done as they are just placeholder values.
        """
        self.timing_data = TimingData('jd', test_epochs, test_mtts, test_mtts_err, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
        time_obj = time.Time(np.array([1.0,1.0, 1.0, 1.0]),format = 'jd',scale = 'utc')
        obj_location = coordinates.SkyCoord(ra = 97.6,dec = 29.67, unit = 'deg')
        obs_location = coordinates.EarthLocation(lat = 43.60, lon = 116.21)
        expected_result = ([1.0,1.0,1.0,1.0])
        actual_result = self.timing_data._calc_barycentric_time(time_obj,obj_location,obs_location)
        self.assertTrue(np.array_equal(expected_result, actual_result))

    def test_bary_time_corrected_time_vals(self):
        """ Successful test to check that the corrected time values are the correct values.

            Within the barycentric time function the corrected time time values are the values that have been fully corrected for the barycentric light correction.
        """
        test_epochs = np.array([0, 294, 298, 573])
        test_mtts = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
        timing_data = TimingData('jd', test_epochs, test_mtts, test_mtts_err, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
        time_obj = time.Time(test_mtts,format = 'jd',scale = 'utc')
        obj_location = coordinates.SkyCoord(ra = 97.6,dec = 29.67, unit = 'deg')
        obs_location = coordinates.EarthLocation(lat = 43.60, lon = -116.21)
        expected_result = np.array([3.11451337e-03,3.20883762e+02,3.25249628e+02,6.25389263e+02])
        actual_result = timing_data._calc_barycentric_time(time_obj, obj_location, obs_location)
        self.assertTrue((np.allclose(expected_result,actual_result, rtol=1e-05, atol=1e-08)))

    def test_ltt_bary(self):
        """ Successful check for the correct barycentric time correction.

            Within the barycentric time function the ltt bary values are the amount needed to adjust the values by for the barycenter.
        """
        time_obj = time.Time(test_mtts, format='jd', scale='utc')
        obj_location = coordinates.SkyCoord(ra=97.6, dec=29.67, unit='deg')
        obs_location = coordinates.EarthLocation(lat=43.60, lon=-116.21)
        time_obj.location = obs_location
        actual_ltt_bary = time_obj.light_travel_time(obj_location).value
        expected_ltt_bary = np.array([0.00274199, 0.00538933, 0.00525568, 0.0038908])
        self.assertTrue((np.allclose(expected_ltt_bary, actual_ltt_bary, rtol=1e-05, atol=1e-08)))

    def test_tra_or_occ_None(self):
        """ Successful test for when the tra_or_occ equals None.

            If there is no tra_or_occ data given an array of 'tra' are produced with the length of the epoch array.
        """
        self.timing_data = TimingData('jd', test_epochs, test_mtts, test_mtts_err, tra_or_occ = None, time_scale='tdb')
        expected_result = np.array(['tra','tra','tra','tra'])
        result = self.timing_data.tra_or_occ
        assert_array_equal(expected_result, result)
    
    def test_only_tra_or_occ_value(self):
        """ Unsuccessful test to check if the tra_or_occ array contains values other than 'tra' or 'occ'.

        """
        not_tra_or_occ = np.array(['tra','occ','trac','oc'])
        with self.assertRaises(ValueError, msg="tra_or_occ array cannot contain string values other than 'tra' or 'occ'"):
             TimingData('jd', test_epochs, test_mtts, test_mtts_err,not_tra_or_occ, time_scale='tdb')  

    def test_tra_or_occ_array(self):
        """ Unsuccessful test to check if the tra_or_occ varible is a numpy array.

        """
        not_tra_or_occ = str(tra_or_occ)
        with self.assertRaises(TypeError, msg= "The variable 'tra_or_occ' expected a NumPy array (np.ndarray) but received a different data type"):
             TimingData('jd', test_epochs, test_mtts, test_mtts_err,not_tra_or_occ, time_scale='tdb')  

    def test_tra_or_occ_shape(self):
        """ Unsuccessful test to check the length of the tra_or_occ array.

            The tra_or_occ array must be the same length as the epochs. mid times and mid time uncertainties arrays.
        """
        not_test_epochs = np.array([0,1])
        not_test_mtts = np.array([0.,1.0,3.0,4.0,5.0])
        not_tra_or_occ = np.array(['occ','tra','occ'])
        with self.assertRaises(ValueError, msg= "Shapes of 'tra_or_occ', 'mid_time_uncertainties', and 'mid_times' arrays do not match."):
             TimingData('jd', not_test_epochs, not_test_mtts, test_mtts_err,not_tra_or_occ, time_scale='tdb')  

    def test_tra_or_occ_no_null(self):
        """ Unsuccessful test to check if any null values are in the tra_or_occ array.

            The data values cannot contain any null values of an error will be rasied if not.
        """
        not_tra_or_occ = np.array(['tra','occ', None, None])
        with self.assertRaises(ValueError, msg = "The 'tra_or_occ' array contains NaN (Not-a-Number) values."):
             TimingData('jd', test_epochs, test_mtts, test_mtts_err,not_tra_or_occ, time_scale='tdb')  
    


    
    # def test_successful_instantiation_jd_no_timescale(self):
    #     transit_times = TransitTimes('jd', )
    # def test_successful_instantiation_jd_non_tdb_timescale(self):
    #     transit_times = TransitTimes('jd', )
    # def test_successful_instantiation_non_jd_tdb_timescale(self):
    #     transit_times = TransitTimes('mjd', time_scale='tdb')
    # def test_successful_instantiation_non_jd_no_timescale(self):
    #     transit_times = TransitTimes('', )
    # def test_successful_instantiation_non_jd_non_tdb_timescale(self):
    #     transit_times = TransitTimes('', )
    # # Test instantiating with ra/dec and without ra/dec vals (and only one val)
    # # Test instantiating
    # def test_no_format(self):
    #     transit_times = TransitTimes()
    # def test_no_obj_coords(self):
    #     transit_times = TransitTimes()
    # def test_all_data_success():
    #     pass

    # def test_all_data_fail():
    #     pass

    # def test_no_uncertainties():
    #     pass
if __name__ == "__main__":
    unittest.main()