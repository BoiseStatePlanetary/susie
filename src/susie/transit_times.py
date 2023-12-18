import numpy as np
from astropy import time
from astropy import coordinates as coord
from astropy import units as u
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
        self.epochs = epochs
        if mid_transit_times_uncertainties is None:
            # Make an array of 1s in the same shape of epochs and mid_transit_times
            mid_transit_times_uncertainties = np.ones_like(self.epochs, dtype=float)
        self.mid_transit_times = mid_transit_times
        self.mid_transit_times_uncertainties = mid_transit_times_uncertainties
        # Check that timing system and scale are JD and TDB
        if time_format != 'jd' or time_scale != 'tdb':
            # If not correct time format and scale, create time objects and run corrections
            logging.warning(f"Recieved time format {time_format} and time scale {time_scale}. " 
                            "Correcting all times to BJD timing system with TDB time scale. If this is incorrect, please set the time format and time scale for TransitTime object.")
            self.mid_transit_times = None
            self.mid_transit_times_uncertainties = None
            mid_transit_times_obj = time.Time(mid_transit_times, format=time_format, scale=time_scale)
            mid_transit_times_uncertainties_obj = time.Time(mid_transit_times_uncertainties, format=time_format, scale=time_scale)
            self._validate_times(mid_transit_times_obj, mid_transit_times_uncertainties_obj, (object_ra, object_dec), (observatory_lon, observatory_lat))
        # Call validation function
        self._validate()

    def _calc_barycentric_time(self, time_obj, obj_location, obs_location):
        """Function to correct non-barycentric time formats to Barycentric Julian Date in TDB time scale."""
        # If given uncertainties, check they are actual values and not placeholders vals of 1
        # If they are placeholder vals, no correction needed, just return array of 1s
        if np.all(time_obj.value == 1):
            return time_obj.value
        time_obj.location = obs_location
        ltt_bary = time_obj.light_travel_time(obj_location)
        corrected_time_vals = (time_obj.tdb+ltt_bary).value
        return corrected_time_vals
    
    def _validate_times(self, mid_transit_times_obj, mid_transit_times_uncertainties_obj, obj_coords, obs_coords):
        """Checks that object and observatory coordinates are in correct format for correction function."""
        # check if there are objects coords, raise error if not
        if all(elem is None for elem in obj_coords):
            raise ValueError("Recieved None for object right ascension and/or declination. " 
                             "Please enter ICRS coordinate values in degrees for object_ra and object_dec for TransitTime object.")
        # Check if there are observatory coords, raise warning and use earth grav center coords if not
        if all(elem is None for elem in obs_coords):
            logging.warning(f"Unable to process observatory coordinates {obs_coords}. "
                             "Using gravitational center of Earth at North Pole.")
            obs_location = coord.EarthLocation.from_geocentric(0., 0., 0., unit=u.m)
        else:
            obs_location = coord.EarthLocation.from_geodetic(obs_coords[0], obs_coords[1])
        obj_location = coord.SkyCoord(ra=obj_coords[0], dec=obj_coords[1], unit='deg', frame='icrs')
        logging.warning(f"Using ICRS coordinates in degrees of RA and Dec {round(obj_location.ra.value, 2), round(obj_location.dec.value, 2)} for time correction. "
                        f"Using geodetic Earth coordinates in degrees of longitude and latitude {round(obs_location.lon.value, 2), round(obs_location.lat.value, 2)} for time correction.")
        # Perform correction, will return array of corrected times
        self.mid_transit_times_uncertainties = self._calc_barycentric_time(mid_transit_times_uncertainties_obj, obj_location, obs_location)
        self.mid_transit_times = self._calc_barycentric_time(mid_transit_times_obj, obj_location, obs_location)

    def _validate(self):
        """Checks that all object attributes are of correct types and within value constraints."""
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