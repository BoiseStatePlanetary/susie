import numpy as np

class TransitTimes(object):
    # TODO: Have user input their timing system, store their original times, if it is not BJD TDB then convert 
    # (will need coords of observatory and coords of star, 
    # can let user not put in coords of observatory and use grav center of Earth)

    """Docstrings for transit times object.
 
    Parameters
    ------------
        epochs : NumPy array
            ints representing ???
        mid_transit_times : NumPy array
            floats representing ??
        Uncertainties : Numpy array
             floats reprensting the uncertainities in ??, has same shape as epochs and mid_transit_times
    Raises
    ------------
        Error raised if parameters are not NumPy Arrays, parameters are not the same shape of array, the values of epochs are not all ints, the values of mid_transit_times and unertainites are not all floats, or values of uncertainities are not all positive.
    """
    def __init__(self, epochs, mid_transit_times, mid_transit_times_uncertainties=None):
        # Default time: maybe BJD?
        # self.time_system = time_system # Should we add a default time system?
        # self.time_system = Time(mid_transit_times, format='time_system') 
        
        self.epochs = epochs
        self.mid_transit_times = mid_transit_times
        self.mid_transit_times_uncertainties = mid_transit_times_uncertainties
        if mid_transit_times_uncertainties is None:
            # Make an array of 1s in the same shape of epochs and mid_transit_times
            self.mid_transit_times_uncertainties = np.ones_like(self.epochs, dtype=float)
        # Call validation function
        self._validate()

    def _validate(self):
        # Check that all are of type array
        if not isinstance(self.epochs, np.ndarray):
            raise TypeError("The variable 'epochs' expected a NumPy array (np.ndarray) but received a different data type")
        if not isinstance(self.mid_transit_times, np.ndarray):
            raise TypeError("The variable 'mid_transit_times' expected a NumPy array (np.ndarray) but received a different data type")
        if not isinstance(self.mid_transit_times_uncertainties, np.ndarray):
            raise TypeError("The variable 'mid_transit_times_uncertainties' expected a NumPy array (np.ndarray) but received a different data type")
        # Check that all are same shape
        if self.epochs.shape != self.mid_transit_times.shape != self.mid_transit_times_uncertainties.shape:
            raise ValueError("Shapes of 'epochs', 'mid_transit_times', and 'mid_transit_times_uncertainties' arrays do not match.")
        # Check that all values in arrays are correct
        if not all(isinstance(value, (int, np.int64)) for value in self.epochs):
            raise TypeError("All values in 'epochs' must be of type int.")
        if not all(isinstance(value, float) for value in self.mid_transit_times):
            raise TypeError("All values in 'mid_transit_times' must be of type float.")
        if not all(isinstance(value, float) for value in self.mid_transit_times_uncertainties):
            raise TypeError("All values in 'mid_transit_times_uncertainties' must be of type float.")
        # Check that there are no null values
        if np.any(np.isnan(self.epochs)):
            raise ValueError("The 'epochs' array contains NaN (Not-a-Number) values.")
        if np.any(np.isnan(self.mid_transit_times)):
            raise ValueError("The 'mid_transit_times' array contains NaN (Not-a-Number) values.")
        if np.any(np.isnan(self.mid_transit_times_uncertainties)):
            raise ValueError("The 'mid_transit_times_uncertainties' array contains NaN (Not-a-Number) values.")
        # Check that mid_transit_times_uncertainties are positive and non-zero
        if not np.all(self.mid_transit_times_uncertainties > 0):
            raise ValueError("The 'mid_transit_times_uncertainties' array must contain non-negative and non-zero values.")
