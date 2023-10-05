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
    def __init__(self, epochs, mid_transit_times, uncertainties=None):
        self.epochs = epochs
        self.mid_transit_times = mid_transit_times
        #self.mid_transit_time_system = 
        self.uncertainties = uncertainties
        if uncertainties is None:
            # Make an array of 1s in the same shape of epochs and mid_transit_times
            self.uncertainties = np.ones_like(self.epochs, dtype=float)
        # Call validation function
        self._validate()

    def _validate(self):
        # Check that all are of type array
        if not isinstance(self.epochs, np.ndarray):
            raise TypeError("The variable 'epochs' expected a NumPy array (np.ndarray) but received a different data type")
        if not isinstance(self.mid_transit_times, np.ndarray):
            raise TypeError("The variable 'mid_transit_times' expected a NumPy array (np.ndarray) but received a different data type")
        if not isinstance(self.uncertainties, np.ndarray):
            raise TypeError("The variable 'uncertainties' expected a NumPy array (np.ndarray) but received a different data type")
        # Check that all are same shape
        if self.epochs.shape != self.mid_transit_times.shape != self.uncertainties.shape:
            raise ValueError("Shapes of 'epochs', 'mid_transit_times', and 'uncertainties' arrays do not match.")
        # Check that all values in arrays are correct
        if not all(isinstance(value, (int, np.int64)) for value in self.epochs):
            raise TypeError("All values in 'epochs' must be of type int.")
        if not all(isinstance(value, float) for value in self.mid_transit_times):
            raise TypeError("All values in 'mid_transit_times' must be of type float.")
        if not all(isinstance(value, float) for value in self.uncertainties):
            raise TypeError("All values in 'uncertainties' must be of type float.")
        # Check that there are no null values
        if np.any(np.isnan(self.epochs)):
            raise ValueError("The 'epochs' array contains NaN (Not-a-Number) values.")
        if np.any(np.isnan(self.mid_transit_times)):
            raise ValueError("The 'mid_transit_times' array contains NaN (Not-a-Number) values.")
        if np.any(np.isnan(self.uncertainties)):
            raise ValueError("The 'uncertainties' array contains NaN (Not-a-Number) values.")
        # Check that uncertainties are positive and non-zero
        if not np.all(self.uncertainties > 0):
            raise ValueError("The 'uncertainties' array must contain non-negative and non-zero values.")


if __name__ == '__main__':
    # CODE BLOCK FOR SMALL TESTS, WILL BE DELETED
    test = np.array([1, 2, 3, 4])
    print(type(test))

    data = np.genfromtxt("./malia_examples/WASP12b_transit_ephemeris.csv", delimiter=',', names=True)
    epochs = data['epoch']
    print(epochs)
    print(type(epochs))

    print(type(epochs) == np.ndarray)
    print(epochs.shape)
    print(data['transit_time'].shape)
    print('Beep')
    print(isinstance(epochs, np.ndarray))
    print(all(isinstance(value, int) for value in epochs))
    print(type(epochs[0]))
    epochs = epochs.astype(int)
    print(epochs)
    print(type(epochs[0]))

    tt1 = TransitTimes(epochs, data['transit_time'], data['sigma_transit_time'])
    print(vars(tt1))