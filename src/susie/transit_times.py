import numpy as np
from astropy import time
from astropy import coordinates as coord
from astropy import units as u
import logging

class TransitTimes(object):
    """Represents transit midpoint data over time. Holds data to be accessed by Ephemeris class.
    
    The transit_times object is a class which formats user data to be passed to the ephemeris.py object. \
        This object creates and/or formats the array of uncertainties in mid_transit_times. This object \
            will also correct user data to use the Barycentric Julian Date as the timing system and Barycentric \
                Dynamical time as the time scale.
    STEP 1: Make an array of 1's to be the uncertainities in the same shape as epochs and mid_transit_times.
    STEP 2: Check that the time system and scale are correct, and if not correct them to be JD and TBD.
    STEP 3: Check that the array's are formatted properly. The appropriate Type or Value Error is raised if there are any issues.
 
    Parameters
    ------------
        time_format: str 
            An abbreviation of the data's timing system. Abbreviations for systems can be found on [Astropy's Time documentation](https://docs.astropy.org/en/stable/time/#id3).
        epochs: numpy.ndarray(int)
            List of reference points for transit observations represented in the transit times data.
        mid_transit_times: numpy.ndarray(float)
            List of observed transit midpoints in days corresponding with epochs.
        mid_transit_times_uncertainties: Optional(numpy.ndarray[float])
            List of uncertainties in days corresponding with transit midpoints. If given None, will be replaced with array of 1's with same shape as `mid_transit_times`.
        time_scale: Optional(str)
            An abbreviation of the data's timing scale. Abbreviations for scales can be found on [Astropy's Time documentation](https://docs.astropy.org/en/stable/time/#id6).
        object_ra: Optional(float)
            The right ascension in degrees of observed object represented by data.
        object_dec: Optional(float)
            The declination in degrees of observed object represented by data.
        observatory_lon: Optional(float)
            The longitude in degrees of observatory data was collected from.
        observatory_lat: Optional(float) 
            The latitude in degrees of observatory data was collected from.
    Raises
    ------------
        Error raised if parameters are not NumPy Arrays, parameters are not the same shape of array, the values of epochs are not all ints, the values of mid_transit_times and unertainites are not all floats, or values of uncertainities are not all positive.

    Side Effects
    -------------
        Epochs and mid transit times are shifted to start at zero by subtracting the minimum number from each value.
    """
    def __init__(self, time_format, epochs, mid_transit_times, mid_transit_times_uncertainties=None, time_scale=None, object_ra=None, object_dec=None, observatory_lon=None, observatory_lat=None):
        # TODO: What would happen here if a non-numpy array is passed in? Should we add validation for that?
        self.epochs = epochs
        self.mid_transit_times = mid_transit_times
        if mid_transit_times_uncertainties is None:
            # Make an array of 1s in the same shape of epochs and mid_transit_times
            mid_transit_times_uncertainties = np.ones_like(self.epochs, dtype=float)
        self.mid_transit_times_uncertainties = mid_transit_times_uncertainties
        # Check that timing system and scale are JD and TDB
        if time_format != 'jd' or time_scale != 'tdb':
            # TODO: Make sure they know default scale is UTC
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
        # Shift epochs and mid transit times
        self.epochs = self.epochs - np.min(self.epochs)
        self.mid_transit_times = self.mid_transit_times - np.min(self.mid_transit_times)

    def _calc_barycentric_time(self, time_obj, obj_location, obs_location):
        """Function to correct non-barycentric time formats to Barycentric Julian Date in TDB time scale.

        STEP 1: Checks if given placeholder values of 1. If given placeholder values, no correction needed and array of 1's returned.

        STEP 2: If given actual values, correct the values to be Barycentric Julian Date in TDB time scale. Return corrected values.

        Parameters
        ----------
            time_obj : numpy.ndarray[float]
                List to be corrected to the Barycentric Julian Date in TDB time scale.
            obj_location : numpy.ndarray[float]
                List of the RA and DEC in degrees of the object being observed.
            obs_location : Optional(numpy.ndarray[float])                           NOTE considered optional only if keyword argument - check for these
                List of the longitude and latitude in degrees of the site of observation. If None given, uses gravitational center of Earth at North Pole.
       
        Returns
        -------
            time_obj.value : numpy.ndarray[float]
                Returned only if no correction needed. Placeholder array of 1s with same shape as `mid_transit_times`.
            corrected_time_vals : numpy.ndarray[float]
                List now corrected to Barycentric Julian Date in TDB time scale.
        """
        # If given uncertainties, check they are actual values and not placeholders vals of 1
        # If they are placeholder vals, no correction needed, just return array of 1s
        if np.all(time_obj.value == 1):
            return time_obj.value
        time_obj.location = obs_location
        ltt_bary = time_obj.light_travel_time(obj_location)
        corrected_time_vals = (time_obj.tdb+ltt_bary).value
        return corrected_time_vals
    
    def _validate_times(self, mid_transit_times_obj, mid_transit_times_uncertainties_obj, obj_coords, obs_coords):
        """Checks that object and observatory coordinates are in correct format for correction function, passes the observed transit midpoints and the uncertainties in observed transit midpoints to the correction function. 
    
        STEP 1: Checks if there are object coordinates (right ascension and declination). Raises a ValueError if not.

        STEP 2: Checks if there are observatory coordinates (latitude and longitude). Raises a warning and uses the gravitational center of Earth at the North Pole if not.

        STEP 3: Performs corrections to 'mid_transit_times_obj' and 'mid_transit_times_uncertainties_obj' by calling '_calc_barycentric_time'

        Parameters
        ----------
            mid_transit_times_obj : (astropy.time.Time[array, string, string])         
                List of transit midpoints, time_format, and time_scale 
            mid_trnasit_times_uncertainties_obj : Optional(astropy.time.Time[array, string, string])      NOTE also not really uncertainties in strings - maybe reformat the description
                List of uncertainties corresponding with transit midpoints, time_format, and time_scale. If given None initailly, have been replaced with array of 1's with same shape as `mid_transit_times`.
            obj_coords : numpy.ndarray[float]
                List of the RA and DEC in degrees of the object being observed.
            obs_coords : Optional(numpy.ndarray[float])             
                List of the longitude and latitude in degrees of the site of observation. If None given, use gravitational center of Earth at North Pole.

        Raises
            ValueError :
                Error if None recieved for object_ra or object_dec.
        ------

        """
        # check if there are objects coords, raise error if not
        if all(elem is None for elem in obj_coords):
            raise ValueError("Recieved None for object right ascension and/or declination. " 
                             "Please enter ICRS coordinate values in degrees for object_ra and object_dec for TransitTime object.")
        # Check if there are observatory coords, raise warning and use earth grav center coords if not
        if all(elem is None for elem in obs_coords):
            logging.warning(f"Unable to process observatory coordinates {obs_coords}. "
                             "Using gravitational center of Earth.")
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
        """Checks that all object attributes are of correct types and within value constraints.
        STEP 1: Check all object attributes are of type array.

        STEP 2: Check all object attributes are of same shape.

        STEP 3: Check all object attributes contain correct value type.

        STEP 4: Check all object attributes contain no null values.

        STEP 5: Check 'mid_transit_times_uncertainties' contains non-negative and non-zero values.

        Raises
        ------
            TypeError :
                Error if 'epochs', 'mid_traisit_times', or 'mid_transit_times_uncertainties' are not NumPy arrays.
            ValueError :
                Error if shapes of 'epochs', 'mid_transit_times', and 'mid_transit_times_uncertainties' arrays do not match.
            TypeError :
                Error if values in 'epochs' are not ints, values in 'mid_transit_times' or 'mid_transit_times_uncertainties" are not floats. 
            ValueError :
                Error if 'epochs', 'mid_transit_times', or 'mid_transit_times_uncertainties' contain a NaN (Not-a-Number) value.
            ValueError :
                Error if 'mid_transit_times_uncertainties' contains a negative or zero value.
        """
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