import numpy as np
import unittest

class assertDictAlmostEqual(unittest.TestCase):
    def __init__(self):
        print("Hi")

        
    def assertDictAlmostEqual(self, d1, d2, msg=None, places=7):
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
