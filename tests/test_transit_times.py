import sys
sys.path.append(".")
from src.susie.transit_times import TransitTimes
import unittest
import numpy as np

# test_epochs = [0, 294, 298, 573, 579, 594, 602, 636, 637, 655, 677, 897, 901, 911, 912, 919, 941, 941, 963, 985, 992, 994, 995, 997, 1015, 1247, 1257, 1258, 1260, 1272, 1287, 1290, 1311, 1312, 1313, 1316, 1317, 1323, 1324, 1333, 1334, 1344, 1345, 1346, 1347, 1357, 1365, 1366, 1585, 1589, 1611, 1619, 1621, 1633, 1637, 1640, 1653, 1661, 1662, 1914, 1915, 1916, 1917, 1937, 1938, 1960, 1964, 1967, 1968, 1969, 1978, 1981, 1991, 1996, 2005, 2012, 2019, 2021, 2022, 2264, 2286, 2288, 2318, 2319, 2331, 2332, 2338, 2339, 2371, 2593, 2634, 2635, 2667, 2668, 2690, 2892, 2910, 2921, 2924, 2942, 2943, 2978, 2979, 2984, 2985, 2988, 2992, 2992, 2997, 2999, 3010, 3017, 3018, 3019, 3217, 3239, 3248, 3260, 3261, 3264, 3306, 3307, 3314, 3316, 3318, 3335, 3335, 3336, 3339, 3341, 3341, 3342, 3342, 3345, 3356, 3570, 3625, 3646, 3657]
# test_mtts = [0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421, 631.933999999892, 648.3059999998659, 657.0360000003129, 694.1440000003204, 695.2370000001974, 714.8820000002161, 738.8940000003204, 979.0049999998882, 983.3710000002757, 994.285000000149, 995.3769999998622, 1003.0160000002943, 1027.0270000002347, 1027.027999999933, 1051.0389999998733, 1075.0509999999776, 1082.691000000108, 1084.8730000001378, 1085.9650000003166, 1088.1480000000447, 1107.7930000000633, 1361.003000000026, 1371.9169999998994, 1373.0079999999143, 1375.191000000108, 1388.2889999998733, 1404.658999999985, 1407.933999999892, 1430.8530000001192, 1431.945000000298, 1433.036000000313, 1436.3100000000559, 1437.4020000002347, 1443.9500000001863, 1445.0419999998994, 1454.8640000000596, 1455.9560000002384, 1466.8700000001118, 1467.9620000002906, 1469.0530000003055, 1470.1450000000186, 1481.058999999892, 1489.7900000000373, 1490.8810000000522, 1729.9020000002347, 1734.2690000003204, 1758.2800000002608, 1767.0109999999404, 1769.194000000134, 1782.2910000002012, 1786.657000000123, 1789.9300000001676, 1804.1189999999478, 1812.851000000257, 1813.942000000272, 2088.9799999999814, 2090.0709999999963, 2091.163000000175, 2092.25400000019, 2114.0819999999367, 2115.1740000001155, 2139.185000000056, 2143.5509999999776, 2146.8250000001863, 2147.916000000201, 2149.0079999999143, 2158.8310000002384, 2162.1049999999814, 2173.0190000003204, 2178.4769999999553, 2188.2990000001155, 2195.939000000246, 2203.5789999999106, 2205.7620000001043, 2206.853000000119, 2470.9769999999553, 2494.9879999998957, 2497.1710000000894, 2529.913000000175, 2531.0049999998882, 2544.1019999999553, 2545.19299999997, 2551.7420000000857, 2552.8330000001006, 2587.7590000000782, 2830.0540000000037, 2874.8020000001416, 2875.8930000001565, 2910.81799999997, 2911.910000000149, 2935.9210000000894, 3156.388000000268, 3176.033999999985, 3188.0389999998733, 3191.313000000082, 3210.9590000002645, 3212.0500000002794, 3250.25, 3251.341000000015, 3256.7990000001155, 3257.8900000001304, 3261.1639999998733, 3265.529000000097, 3265.530999999959, 3270.9870000001974, 3273.1699999999255, 3285.1750000002794, 3292.814999999944, 3293.907000000123, 3294.998000000138, 3511.098999999929, 3535.1099999998696, 3544.933999999892, 3558.0300000002608, 3559.121999999974, 3562.3960000001825, 3608.2349999998696, 3609.3270000000484, 3616.966000000015, 3619.149999999907, 3621.3330000001006, 3639.885000000242, 3639.8870000001043, 3640.978000000119, 3644.253000000026, 3646.435000000056, 3646.435000000056, 3647.526000000071, 3647.526000000071, 3650.8009999999776, 3662.805999999866, 3896.3700000001118, 3956.3980000000447, 3979.31799999997, 3991.323000000324]
# test_mtts_err = [0.00043, 0.00028, 0.00062, 0.00042, 0.00043, 0.00032, 0.00036, 0.00046, 0.00041, 0.00019, 0.00043, 0.00072, 0.00079, 0.00037, 0.00031, 0.0004, 0.0004, 0.00028, 0.00028, 0.00068, 0.00035, 0.00029, 0.00024, 0.00029, 0.00039, 0.00027, 0.00021, 0.00027, 0.00024, 0.00032, 0.00031, 0.00022, 0.00018, 0.00017, 0.00033, 0.00011, 0.0001, 0.00017, 0.00032, 0.00039, 0.00035, 0.00034, 0.00035, 0.00032, 0.00042, 0.00037, 0.00037, 0.00031, 0.00033, 0.00039, 0.0003, 0.0003, 0.0003, 0.0003, 0.00046, 0.00024, 0.00038, 0.00027, 0.00029, 0.00021, 0.0003, 0.00033, 0.00071, 0.00019, 0.00043, 0.00034, 0.00034, 0.00019, 0.00019, 0.00031, 0.00028, 0.00032, 0.0004, 0.00029, 0.00029, 0.00025, 0.00034, 0.00034, 0.00046, 0.00043, 0.00039, 0.00049, 0.00046, 0.00049, 0.00035, 0.00036, 0.00022, 0.0002, 0.00031, 0.00042, 0.00033, 0.00033, 0.00055, 0.00023, 0.00021, 0.00035, 0.00025, 0.00034, 0.00037, 0.00028, 0.00023, 0.00028, 0.00039, 0.00024, 0.00022, 0.00029, 0.00043, 0.00036, 0.00026, 0.00048, 0.00032, 0.0004, 0.00018, 0.00021, 0.00056, 0.00023, 0.0003, 0.00022, 0.00034, 0.00028, 0.00027, 0.00035, 0.00031, 0.00032, 0.00033, 0.0005, 0.00031, 0.00032, 0.00091, 0.00035, 0.00026, 0.00021, 0.00034, 0.00034, 0.00038, 0.0004, 0.00026, 0.0003, 0.00044]
test_epochs = np.array([0, 294, 298, 573])
test_mtts = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
test_mtts_err = np.array([0.00043, 0.00028, 0.00062, 0.00042])
class TestTransitTimes(unittest.TestCase):
    """
    Tests:
    beep
    ** s = successful, us = unsuccessful
        test s that each variable is of np.ndarray type=done
        test us that each variable is of np.ndarray type=done
        test s that values in each array are of specified type (epochs=ints, mid_transit_times=floats, uncertainties=floats)=done
        test us that values in each array are of specified type (epochs=ints, mid_transit_times=floats, uncertainties=floats)=done
        test s that all variables have same shape= done
        test us that all variables have same shape=done
        test s that there are no null/nan values=done
        test us that there are no null/nan values=done
        test s that uncertainties are all non-negative and non-zero=done
        test s creation of uncertainties if not given=done
    TODO:
        set up and tear down for transit times=done
        successful 0, neg and positive =done
        epochs - type of variable (np.array), type of values (int), values are what u expect (if u pass in starting at 0, >0, <0)
        mid transit times - type of variable (np.array), type of values (float), values are what u expect (if u pass in starting at 0, >0, <0)
        mid transit time uncertainties - type of var (np.array), type of vals (float), values are what u expect (if pass in None, array of ones, else (if you pass in actual data and not None) data you pass in
        midtransit times are non-neg
    """

    # Set Up and Tear down Transit times
   
       
    # Test instantiating with correct and incorrect timescales
    def test_successful_instantiation_jd_tdb_timescale(self):
        """Successful creation of the TimingData object with proper timescale

            Creating the TimingData object with the proper timescale of JD TDB which is
            barycentric Julian Date. Also including uncertainties.
        """
        # Should not get any errors, the epochs and transit times should be the same as they are inputted
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertIsInstance(self.transit_times, TransitTimes)  # Check if the object is an instance of TransitTimes
        shifted_epochs = test_epochs - np.min(test_epochs)
        self.assertTrue(np.array_equal(self.transit_times.epochs, shifted_epochs))  # Check if epochs remain unchanged
        self.assertTrue(np.array_equal(self.transit_times.mid_transit_times, test_mtts))  # Check mid_transit_times
        self.assertTrue(np.array_equal(self.transit_times.mid_transit_times_uncertainties, test_mtts_err))  # Check uncertainties

    def test_s_init_jd_tdb_no_uncertainties(self):
        # Should not get any errors, the epochs and transit times should be the same as they are inputted
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts, time_scale='tdb')
        self.assertIsInstance(self.transit_times, TransitTimes)  # Check if the object is an instance of TransitTimes
        shifted_epochs = test_epochs - np.min(test_epochs)
        shifted_mtt = test_mtts - np.min(test_mtts)
        new_uncertainties = np.ones_like(test_epochs, dtype=float)
        self.assertTrue(np.array_equal(self.transit_times.epochs, shifted_epochs))  # Check if epochs remain unchanged
        self.assertTrue(np.array_equal(self.transit_times.mid_transit_times, shifted_mtt))  # Check mid_transit_times
        self.assertTrue(np.array_equal(self.transit_times.mid_transit_times_uncertainties, new_uncertainties))  # Check uncertainties chage this back!!!

    

    # Test instantiating with correct and incorrect timescales


    # Tests for numpy array validation
    def test_us_epochs_arr_type_str(self):
        # epochs are strings instead of numpy array
        string_test_epochs_arr = str(test_epochs)
        with self.assertRaises(TypeError, msg="The variable 'epochs' expected a NumPy array (np.ndarray) but received a different data type"):
            TransitTimes('jd', string_test_epochs_arr, test_mtts, test_mtts_err, time_scale='tdb')
     
    def test_us_mtts_arr_type_str(self):
        # midtransitstimes are strings instead of numpy array
        string_test_mtts_arr = str(test_mtts)
        with self.assertRaises(TypeError, msg="The variable 'mid_transit_times' expected a NumPy array (np.ndarray) but received a different data type"):
            TransitTimes('jd', test_epochs, string_test_mtts_arr, test_mtts_err, time_scale='tdb')
    
    def test_us_mtts_err_arr_type_str(self):
        # mid transit times uncertainites are strings instead of numpy array
        string_test_mtts_err_arr = str(test_mtts_err)
        with self.assertRaises(TypeError, msg="The variable 'mid_transit_times_uncertainties' expected a NumPy array (np.ndarray) but received a different data type"):
            TransitTimes('jd', test_epochs, test_mtts, string_test_mtts_err_arr, time_scale='tdb')


    # Test for data value type validation
    def test_s_vars_value_types(self):
        # should not get any errors
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertTrue(all(isinstance(value, (int, np.int64)) for value in self.transit_times.epochs))
        self.assertTrue(all(isinstance(value, float) for value in self.transit_times.mid_transit_times))
        self.assertTrue(all(isinstance(value, float) for value in self.transit_times.mid_transit_times_uncertainties))
   
    def test_us_epochs_value_types_float(self):
        # epochs are floats 
        float_test_epochs = test_epochs.astype(float)
        with self.assertRaises(TypeError, msg="All values in 'epochs' must be of type int."):
            TransitTimes('jd', float_test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
    
    def test_us_mtts_value_types_int(self):
        # mid transit times are int error is not being raised
        int_test_mtts= test_mtts.astype(int)
        with self.assertRaises(TypeError, msg="All values in 'mid_transit_times' must be of type float."):
            TransitTimes('jd', test_epochs, int_test_mtts, test_mtts_err, time_scale='tdb')
    
    def test_us_mtts_err_value_types_int(self):
        # mid transit times uncertanties are int
        int_test_mtts_err= test_mtts_err.astype(int)
        with self.assertRaises(TypeError, msg="All values in 'mid_transit_times_uncertainties' must be of type float."):
            TransitTimes('jd', test_epochs, test_mtts, int_test_mtts_err, time_scale='tdb')


    # Checks that epochs work with positive, negative and 0 values when shifted in validation
    def test_shifted_epochs_zero(self):
        # shifted epochs work with 0 value
        test_epochs_zero = np.array([0, 294, 298, 573]).astype(int)
        shifted_epochs_zero = np.array([0, 294, 298, 573]).astype(int)
        self.transit_times = TransitTimes('jd', test_epochs_zero, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.transit_times.epochs, shifted_epochs_zero))
    
    def test_shifted_epochs_pos(self):
        # shifted epochs work with postive values
        test_epochs_pos = np.array([1, 294, 298, 573]).astype(int)
        shifted_epochs_pos = np.array([0, 293, 297, 572]).astype(int)
        self.transit_times = TransitTimes('jd', test_epochs_pos, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.transit_times.epochs, shifted_epochs_pos))

    def test_shifted_epochs_neg(self):
        # shifted epochs work with negative values
        test_epochs_neg = np.array([-1, 294, 298, 573]).astype(int)
        shifted_epochs_neg = np.array([0, 295, 299, 574]).astype(int)
        self.transit_times = TransitTimes('jd', test_epochs_neg, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.transit_times.epochs, shifted_epochs_neg))


    # Checks that mid transit times work with positive, negative and 0 values when shifted in validation
    def test_shifted_mtts_zero(self):
        #shifted mtts work with 0
        test_mtts_zero = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
        shifted_mtts_zero = np.array([0.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts_zero, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.transit_times.mid_transit_times, shifted_mtts_zero))

    def test_shifted_mtts_pos(self):
        #shifted mtts work with 1
        test_mtts_pos = np.array([1.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
        shifted_mtts_pos = np.array([0.0, 319.8780000000261, 324.24399999994785, 624.3850000002421])
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts_pos, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.transit_times.mid_transit_times, shifted_mtts_pos))
    
    def test_shifted_mtts_neg(self):
        #shifted mtts work with -1
        test_mtts_neg = np.array([-1.0, 320.8780000000261, 325.24399999994785, 625.3850000002421])
        shifted_mtts_neg = np.array([0.0, 321.8780000000261, 326.24399999994785, 626.3850000002421])
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts_neg, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.transit_times.mid_transit_times, shifted_mtts_neg))

#<————————————————————————————————————————————————————————————————————————————————————————>
    #mid transit times uncertainties different arrays
    def test_no_mtts_err(self):
        # mid transit time errors are none
        test_mtts_err = None
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        if test_mtts_err is None:
            new_uncertainities= np.ones_like(test_epochs,dtype=float)
            self.assertTrue(np.all(new_uncertainities==np.ones_like(test_epochs,dtype=float)))
        
    def test_mid_transit_err_ones(self):
        # mid transit time errors are ones
        new_test_mtts_err=np.ones_like(test_mtts_err)
        self.transit_times=TransitTimes('jd', test_epochs, test_mtts, new_test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.transit_times.mid_transit_times_uncertainties,new_test_mtts_err))

    def test_mid_transit_err_neg(self):
        # mid transit time errors are negative
      test_mtts_err_neg= np.array([-0.00043, -0.00028, -0.00062, -0.00042])
      with self.assertRaises(ValueError, msg="The 'mid_transit_times_uncertainties' array must contain non-negative and non-zero values."):
            TransitTimes('jd', test_epochs, test_mtts, test_mtts_err_neg, time_scale='tdb')  
      
    
    def test_mid_transit_err_zero(self):
        # mid transit time errors are zero
      test_mtts_err_zero= np.array([0.,0.,0.,0.])
      with self.assertRaises(ValueError, msg="The 'mid_transit_times_uncertainties' array must contain non-negative and non-zero values."):
            TransitTimes('jd', test_epochs, test_mtts, test_mtts_err_zero, time_scale='tdb')  

    def test_mid_transit_err_self(self):
        # if the data is good then returns the same data
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertTrue(np.array_equal(self.transit_times.mid_transit_times_uncertainties, test_mtts_err))

    #variables have the same shape
    def test_variable_shape(self):
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertEqual(test_epochs.shape, test_mtts.shape, test_mtts_err.shape)

    #variables do not have the same shape
    def test_variable_shape_fail(self):
        new_test_epochs= np.array([0, 298, 573])  
        new_test_mtts= np.array([0.0, 625.3850000002421])
        with self.assertRaises(ValueError, msg="Shapes of 'epochs', 'mid_transit_times', and 'mid_transit_times_uncertainties' arrays do not match."):
            TransitTimes('jd', new_test_epochs, new_test_mtts, test_mtts_err, time_scale='tdb')  
    
   #successful no NaN values in variables
    def successful_no_nan_values(self):
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, time_scale='tdb')
        self.assertNotIn(np.nan,test_epochs)
        self.assertNotIn(np.nan,test_mtts)
        self.assertNotIn(np.nan,test_mtts_err)


    #mid transit times have no NaN values
    def test_mtts_nan(self):
        new_test_mtts=np.array([0., np.nan , 298. ,573.], dtype=float)
        with self.assertRaises(ValueError, msg="The 'mid_transit_times' array contains NaN (Not-a-Number) values."):
            TransitTimes('jd', test_epochs, new_test_mtts, test_mtts_err, time_scale='tdb')  
    
    
     #mid transit time uncertainites have no NaN values
    def test_mtts_err_nan(self):
        new_test_mtts_err=np.array([0.00043, np.nan, 0.00062, 0.00042], dtype=float)
        with self.assertRaises(ValueError, msg="The 'mid_transit_times_uncertainties' array contains NaN (Not-a-Number) values."):
            TransitTimes('jd', test_epochs, test_mtts, new_test_mtts_err, time_scale='tdb')  

    #tests for calc_barycentric_time
    test_time_obj_ones=np.array([1.0, 1.0, 1.0, 1.0])
    test_time_obj=np.array([0.00034,0.0006,0.0005,0.0008])
    test_obj_location= np.array([1.0,2.0])
    test_obs_locations=np.array([2.0,3.0])
    #check uncertainties arent ones
    def calc_bary_time_instantiation(self):
        self.transit_times = TransitTimes('jd', test_epochs, test_mtts, test_mtts_err, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
        self.assertIsInstance(self.transit_times, TransitTimes)
    
    def calc_bary_time_uncertinties(self):
       test_mtts_err=np.array([1.0, 1.0, 1.0, 1.0])
       self.transit_times= TransitTimes('jd', test_epochs, test_mtts,test_mtts_err, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    
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