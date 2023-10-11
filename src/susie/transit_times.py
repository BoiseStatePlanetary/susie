import numpy as np

class TransitTimes(object):
    """Docstrings for transit times object.
 
    Parameters
    ----------
    self : 
    epochs : NumPy array
        what does it do
    mid_transit_times : NumPy
        what does it do 

    Exceptions
    ----------
        lots to add
    """
    def __init__(self, epochs, mid_transit_times, mid_transit_times_uncertainties=None):
        # Default time: maybe BJD?

        # self.time_system = time_system # Should we add a default time system?
        # self.time_system = Time(mid_transit_times, format='time_system') 
        # What is the goal of putting into time obj? Do we want to just make sure 
        # they're aware of the time system they're using or do we eventually want 
        # to be able to use/convert the times in some way?
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
