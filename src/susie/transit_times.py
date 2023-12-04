import numpy as np
from astropy import time
from astropy import coordinates as coord
import logging

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
    def __init__(self, time_format, epochs, mid_transit_times, mid_transit_times_uncertainties=None, time_scale=None, object_ra=None, object_dec=None, observatory_lon=None, observatory_lat=None):
        # Default time: BJD TDB
        self.time = time.Time(mid_transit_times, format=time_format, scale=time_scale)
        self.obj_coords = (object_ra, object_dec)
        self.observatory_coords = (observatory_lon, observatory_lat)
        # TODO: Check if time system and scale are BJD and TDB, if not run a correction 
        self.epochs = epochs
        self.mid_transit_times = mid_transit_times
        self.mid_transit_times_uncertainties = mid_transit_times_uncertainties
        if mid_transit_times_uncertainties is None:
            # Make an array of 1s in the same shape of epochs and mid_transit_times
            self.mid_transit_times_uncertainties = np.ones_like(self.epochs, dtype=float)
        # Call validation function
        self._validate()

    def _calc_barycentric_time(self):
        obj_coord = coord.SkyCoord(self.obj_coords[0], self.obj_coords[1], unit='deg', frame='icrs')
        obs_location = coord.EarthLocation.from_geodetic(self.observatory_coords[0], self.observatory_coords[1])
        self.time.location = obs_location
        time_corrected = self.time.light_travel_time(obj_coord)
        print(time_corrected)

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
        # Check that timing system and scale are JD and TDB
        if self.time.format != 'jd' or self.time.scale != 'tdb':
            # if correction has to happen, raise warning so they know
            logging.warning(f"Recieved time format {self.time.format} and time scale {self.time.scale}. " 
                            "Correcting all times to BJD timing system with TDB time scale. "
                            "If this is incorrect, please set the time format and time scale for TransitTime object.")
            # check if there is skycoord of object and site coordinates of obs
            if all(elem is None for elem in self.obj_coords):
                raise ValueError("Recieved None for object right ascension and/or declination. " 
                                 "Please enter ICRS coordinate values in degrees for object_ra and object_dec for TransitTime object.")
            else:
                # Check if there is observatory coords, if not raise error and use earth coords
                if all(elem is None for elem in self.observatory_coords):
                    logging.warning(f"Unable to process observatory coordinates {self.observatory_coords}. "
                                    "Will use gravitational center of Earth.")
                    # TODO enter grav center of earth coords here
                    # self.observatory_coords = ()
                logging.warning(f"Using ICRS coordinates in degrees of RA and Dec {self.obj_coords} for time correction. "
                                f"Using geodetic Earth coordinates in degrees of longitude and latitude {self.observatory_coords} for time correction.")
                self._calc_barycentric_time()
