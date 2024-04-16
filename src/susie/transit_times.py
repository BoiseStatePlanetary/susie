import numpy as np
from astropy import time
from astropy import coordinates as coord
from astropy import units as u
import logging


class TransitTimes():
    """Represents the pre-processed transit mid point timing data over observations.
    
    Parameters
    ------------
        epochs: numpy.ndarray[int]
            List of orbit number reference points for transit timing observations
        mid_times: numpy.ndarray[float]
            List of observed transit timing mid points corresponding with epochs
        mid_time_uncertainties: numpy.ndarray[float]
            List of uncertainties corresponding to transit mid-times
    """
    def __init__(self, epochs, mid_times, mid_time_uncertainties):
        self.epochs = epochs
        self.mid_times = mid_times
        self.mid_time_uncertainties = mid_time_uncertainties

class OccultationTimes():
    """Represents the pre-processed occultation mid point timing data over observations.
    
    Parameters
    ------------
        epochs: numpy.ndarray[int]
            List of orbit number reference points for occultation timing observations
        mid_times: numpy.ndarray[float]
            List of observed occultation timing mid points corresponding with epochs
        mid_time_uncertainties: numpy.ndarray[float]
            List of uncertainties corresponding to occultation mid-times
    """
    def __init__(self, epochs, mid_times, mid_time_uncertainties):
        self.epochs = epochs
        self.mid_times = mid_times
        self.mid_time_uncertainties = mid_time_uncertainties

class TimingData():
    """Represents timing mid point data over observations. Holds data to be accessed by Ephemeris class.

    The TimingData object processes, formats, and holds user data to be passed to the ephemeris object.
    
    Timing conversions are applied to ensure that all data is processed correctly and users are aware of 
    timing formats and scales, which can give rise to false calculations in our metrics. If data is not specified
    to be in Barycentric Julian Date format with the TDB time scale, timing data will be corrected for barycentric
    light travel time using the Astropy Time utilities. 
    
    If mid time uncertainties are not provided, we will generate placeholder values of 1.

    Our implementations rely on Numpy functions. This object implements checks to ensure that data are stored in 
    Numpy arrays and are of correct data types. The appropriate Type or Value Error is raised if there are any issues.
    
    If timing data contains both transit mid times and occultation mid times, users can pass in an array of 'tra' 
    and 'occ' strings that correspond to the epochs, mid time, and uncertainty timing data. If passed in, timing data
    will be separated according to the order of the tra_or_occ array and stored in their corresponding objects.

    Parameters
    ------------
        time_format: str 
            An abbreviation of the data's time system. Abbreviations for systems can be found on [Astropy's Time documentation](https://docs.astropy.org/en/stable/time/#id3).
        epochs: numpy.ndarray[int]
            List of orbit number reference points for timing observations
        mid_times: numpy.ndarray[float]
            List of observed timing mid points corresponding with epochs, in timing units given by time_format
        mid_time_uncertainties: Optional(numpy.ndarray[float])
            List of uncertainties corresponding to timing mid points, in timing units given by time_format. If given None, will be replaced with array of 1's with same shape as `mid_times`.
        time_scale: Optional(str)
            An abbreviation of the data's time scale. Abbreviations for scales can be found on [Astropy's Time documentation](https://docs.astropy.org/en/stable/time/#id6).
        object_ra: Optional(float)
            The right ascension of observed object represented by data, in degrees
        object_dec: Optional(float)
            The declination of observed object represented by data, in degrees
        observatory_lon: Optional(float)
            The longitude of observatory data was collected from, in degrees
        observatory_lat: Optional(float) 
            The latitude of observatory data was collected from, in degrees
    
    Raises
    ------
        Error raised if  
            * parameters are not NumPy Arrays
            * timing data arrays are not the same shape
            * the values of epochs are not all ints
            * the values of mid_times and uncertainites are not all floats
            * values of uncertainities are not all positive
            * values of transit or occultation array or not all 'tra' or 'occ'

    Side Effects
    -------------
        Variables epochs and mid_times are shifted to start at zero by subtracting the minimum number from each value
    """
    def __init__(self, time_format, epochs, mid_times, mid_time_uncertainties=None, tra_or_occ=None, time_scale=None, object_ra=None, object_dec=None, observatory_lon=None, observatory_lat=None):
        # In ephemeris, we would probably check if we had both transits and occultations
        # If so, we would need to account for that (maybe just in the fitting implementation? anywhere else?)
        # If not, then we just assign our data to the transit times data and let it run!
        # Although, need to look into this still. Might just have this implemented in model fits and plotting?
        self.transits = None
        self.occultations = None
        self.epochs = epochs
        self.mid_times = mid_times
        if mid_time_uncertainties is None:
            # If no uncertainties provided, make an array of 1s in the same shape of epochs
            mid_time_uncertainties =  np.ones_like(self.epochs, dtype=float)
        self.mid_time_uncertainties = mid_time_uncertainties
        # Check that timing system and scale are JD and TDB
        if time_format != 'jd' or time_scale != 'tdb':
            # If not correct time format and scale, create time objects and run corrections
            logging.warning(f"Recieved time format {time_format} and time scale {time_scale}. " 
                            "Correcting all times to BJD timing system with TDB time scale. \
                             If no time scale is given, default is automatically assigned to UTC. \
                             If this is incorrect, please set the time format and time scale for TransitTime object.")
            # Set timing data to None for now
            self.mid_times = None
            self.mid_time_uncertainties = None
            mid_times_obj = time.Time(mid_times, format=time_format, scale=time_scale)
            mid_time_uncertainties_obj = time.Time(mid_time_uncertainties, format=time_format, scale=time_scale)
            self._validate_times(mid_times_obj, mid_time_uncertainties_obj, (object_ra, object_dec), (observatory_lon, observatory_lat))
        # Call validation function
        self._validate()
        # Once everything is validated and corrected, we can separate into transits and occultations if we are given the tra_or_occ data
        if tra_or_occ is None:
            # All data are transit data
            self.transits = TransitTimes(self.epochs, self.mid_times, self.mid_time_uncertainties)
        else:
            tra_or_occ = np.char.strip(tra_or_occ) # Strip any whitespace
            # Check if any values are not valid in tra_or_occ array
            if any(val not in ['tra', 'occ'] for val in tra_or_occ):
                raise ValueError("tra_or_occ array cannot contain string values other than 'tra' or 'occ'")
            # Separate epochs, mid times, and uncertainties into their respective lists
            t_epochs = np.array([self.epochs[i] for i in range(len(tra_or_occ)) if tra_or_occ[i] == 'tra'])
            t_mid_times = np.array([self.mid_times[i] for i in range(len(tra_or_occ)) if tra_or_occ[i] == 'tra'])
            t_mid_time_uncertainties = np.array([self.mid_time_uncertainties[i] for i in range(len(tra_or_occ)) if tra_or_occ[i] == 'tra'])
            o_epochs = np.array([self.epochs[i] for i in range(len(tra_or_occ)) if tra_or_occ[i] == 'occ'])
            o_mid_times = np.array([self.mid_times[i] for i in range(len(tra_or_occ)) if tra_or_occ[i] == 'occ'])
            o_mid_time_uncertainties = np.array([self.mid_time_uncertainties[i] for i in range(len(tra_or_occ)) if tra_or_occ[i] == 'occ'])
            # Create transit and occultation objects
            self.transits = TransitTimes(t_epochs, t_mid_times, t_mid_time_uncertainties)
            self.occultations = OccultationTimes(o_epochs, o_mid_times, o_mid_time_uncertainties)

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
                Returned only if no correction needed. Placeholder array of 1s with same shape as `mid_times`.
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
    
    def _validate_times(self, mid_times_obj, mid_time_uncertainties_obj, obj_coords, obs_coords):
        """Checks that object and observatory coordinates are in correct format for correction function, passes the observed transit midpoints and the uncertainties in observed transit midpoints to the correction function. 
    
        STEP 1: Checks if there are object coordinates (right ascension and declination). Raises a ValueError if not.

        STEP 2: Checks if there are observatory coordinates (latitude and longitude). Raises a warning and uses the gravitational center of Earth at the North Pole if not.

        STEP 3: Performs corrections to 'mid_times_obj' and 'mid_time_uncertainties_obj' by calling '_calc_barycentric_time'

        Parameters
        ----------
            mid_times_obj : (astropy.time.Time[array, string, string])         
                List of transit midpoints, time_format, and time_scale 
            mid_trnasit_times_uncertainties_obj : Optional(astropy.time.Time[array, string, string])      NOTE also not really uncertainties in strings - maybe reformat the description
                List of uncertainties corresponding with transit midpoints, time_format, and time_scale. If given None initailly, have been replaced with array of 1's with same shape as `mid_times`.
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
        self.mid_time_uncertainties = self._calc_barycentric_time(mid_time_uncertainties_obj, obj_location, obs_location)
        self.mid_times = self._calc_barycentric_time(mid_times_obj, obj_location, obs_location)

    def _validate(self):
        """Checks that all object attributes are of correct types and within value constraints.
        STEP 1: Check all object attributes are of type array.

        STEP 2: Check all object attributes are of same shape.

        STEP 3: Check all object attributes contain correct value type.

        STEP 4: Check all object attributes contain no null values.

        STEP 5: Check 'mid_time_uncertainties' contains non-negative and non-zero values.

        Raises
        ------
            TypeError :
                Error if 'epochs', 'mid_traisit_times', or 'mid_time_uncertainties' are not NumPy arrays.
            ValueError :
                Error if shapes of 'epochs', 'mid_times', and 'mid_time_uncertainties' arrays do not match.
            TypeError :
                Error if values in 'epochs' are not ints, values in 'mid_times' or 'mid_time_uncertainties" are not floats. 
            ValueError :
                Error if 'epochs', 'mid_times', or 'mid_time_uncertainties' contain a NaN (Not-a-Number) value.
            ValueError :
                Error if 'mid_time_uncertainties' contains a negative or zero value.
        """
        # Check that all are of type array
        if not isinstance(self.epochs, np.ndarray):
            raise TypeError("The variable 'epochs' expected a NumPy array (np.ndarray) but received a different data type")
        if not isinstance(self.mid_times, np.ndarray):
            raise TypeError("The variable 'mid_times' expected a NumPy array (np.ndarray) but received a different data type")
        if not isinstance(self.mid_time_uncertainties, np.ndarray):
            raise TypeError("The variable 'mid_time_uncertainties' expected a NumPy array (np.ndarray) but received a different data type")
        # Check that all are same shape
        if self.epochs.shape != self.mid_times.shape != self.mid_time_uncertainties.shape:
            raise ValueError("Shapes of 'epochs', 'mid_times', and 'mid_time_uncertainties' arrays do not match.")
        # Check that all values in arrays are correct
        # if not all(isinstance(value, (int, np.int64)) for value in self.epochs) or not all(isinstance(value, (int, np.int32)) for value in self.epochs):
        if not all(isinstance(value, (int, np.int64, np.int32)) for value in self.epochs):
            raise TypeError("All values in 'epochs' must be of type int, numpy.int64, or numpy.int32.")
        if not all(isinstance(value, float) for value in self.mid_times):
            raise TypeError("All values in 'mid_times' must be of type float.")
        if not all(isinstance(value, float) for value in self.mid_time_uncertainties):
            raise TypeError("All values in 'mid_time_uncertainties' must be of type float.")
        # Check that there are no null values
        if np.any(np.isnan(self.epochs)):
            raise ValueError("The 'epochs' array contains NaN (Not-a-Number) values.")
        if np.any(np.isnan(self.mid_times)):
            raise ValueError("The 'mid_times' array contains NaN (Not-a-Number) values.")
        if np.any(np.isnan(self.mid_time_uncertainties)):
            raise ValueError("The 'mid_time_uncertainties' array contains NaN (Not-a-Number) values.")
        # Check that mid_time_uncertainties are positive and non-zero (greater than zero)
        if not np.all(self.mid_time_uncertainties > 0):
            raise ValueError("The 'mid_time_uncertainties' array must contain non-negative and non-zero values.")
        if self.epochs[0] != 0:
            # Shift epochs and mid transit times
            self.epochs -= np.min(self.epochs)
            # TODO import warning that we are minimizing their epochs and transit times
        if self.mid_times[0] != 0:
            self.mid_times -= np.min(self.mid_times)








# class TransitTimes():
#     """Represents transit midpoint data over time. Holds data to be accessed by Ephemeris class.
    
#     The transit_times object is a class which formats user data to be passed to the ephemeris.py object. \
#         This object creates and/or formats the array of uncertainties in mid_transit_times. This object \
#             will also correct user data to use the Barycentric Julian Date as the timing system and Barycentric \
#                 Dynamical time as the time scale.

#     STEP 1: Make an array of 1's to be the uncertainities in the same shape as epochs and mid_transit_times.

#     STEP 2: Check that the time system and scale are correct, and if not correct them to be JD and TBD.

#     STEP 3: Check that the array's are formatted properly. The appropriate Type or Value Error is raised if there are any issues.
 
#     Parameters
#     ------------
#         time_format: str 
#             An abbreviation of the data's timing system. Abbreviations for systems can be found on [Astropy's Time documentation](https://docs.astropy.org/en/stable/time/#id3).
#         epochs: numpy.ndarray(int)
#             List of reference points for transit observations represented in the transit times data.
#         mid_transit_times: numpy.ndarray(float)
#             List of observed transit midpoints in days corresponding with epochs.
#         mid_transit_times_uncertainties: Optional(numpy.ndarray[float])
#             List of uncertainties in days corresponding with transit midpoints. If given None, will be replaced with array of 1's with same shape as `mid_transit_times`.
#         time_scale: Optional(str)
#             An abbreviation of the data's timing scale. Abbreviations for scales can be found on [Astropy's Time documentation](https://docs.astropy.org/en/stable/time/#id6).
#         object_ra: Optional(float)
#             The right ascension in degrees of observed object represented by data.
#         object_dec: Optional(float)
#             The declination in degrees of observed object represented by data.
#         observatory_lon: Optional(float)
#             The longitude in degrees of observatory data was collected from.
#         observatory_lat: Optional(float) 
#             The latitude in degrees of observatory data was collected from.
#     Raises
#     ------
#         Error raised if  
#             * parameters are not NumPy Arrays 
#             * parameters are not the same shape of array 
#             * the values of epochs are not all ints
#             * the values of mid_transit_times and unertainites are not all floats
#             * values of uncertainities are not all positive.

#     Side Effects
#     -------------
#         Variables epochs and mid_transit_times are shifted to start at zero by subtracting the minimum number from each value.
#     """
#     # def __init__(self, time_format, epochs, mid_transit_times, mid_transit_times_uncertainties=None, tra_or_occ=None, time_scale=None, object_ra=None, object_dec=None, observatory_lon=None, observatory_lat=None):
#     def __init__(self, time_format, epochs, mid_transit_times, mid_transit_times_uncertainties=None, time_scale=None, object_ra=None, object_dec=None, observatory_lon=None, observatory_lat=None):
#         self.epochs = epochs
#         self.mid_transit_times = mid_transit_times
#         if mid_transit_times_uncertainties is None:
#             # Make an array of 1s in the same shape of epochs and mid_transit_times
#             mid_transit_times_uncertainties =  np.ones_like(self.epochs, dtype=float)
#         self.mid_transit_times_uncertainties = mid_transit_times_uncertainties
#         # Check that timing system and scale are JD and TDB
#         if time_format != 'jd' or time_scale != 'tdb':
#             # TODO: Make sure they know default scale is UTC
#             # If not correct time format and scale, create time objects and run corrections
#             logging.warning(f"Recieved time format {time_format} and time scale {time_scale}. " 
#                             "Correcting all times to BJD timing system with TDB time scale. If this is incorrect, please set the time format and time scale for TransitTime object.")
#             self.mid_transit_times = None
#             self.mid_transit_times_uncertainties = None
#             mid_transit_times_obj = time.Time(mid_transit_times, format=time_format, scale=time_scale)
#             mid_transit_times_uncertainties_obj = time.Time(mid_transit_times_uncertainties, format=time_format, scale=time_scale)
#             self._validate_times(mid_transit_times_obj, mid_transit_times_uncertainties_obj, (object_ra, object_dec), (observatory_lon, observatory_lat))
#         # Call validation function
#         self._validate()
#         # Once everything is validated and corrected, we can separate into transits and occultations if we are given the tra_or_occ data
#         # if tra_or_occ is not None:
#         #     for epoch, mtt, mtt_err in zip(self.epochs, self.mid_transit_times, self.mid_transit_times_uncertainties):
#         #         foiwehf

#     def _calc_barycentric_time(self, time_obj, obj_location, obs_location):
#         """Function to correct non-barycentric time formats to Barycentric Julian Date in TDB time scale.

#         STEP 1: Checks if given placeholder values of 1. If given placeholder values, no correction needed and array of 1's returned.

#         STEP 2: If given actual values, correct the values to be Barycentric Julian Date in TDB time scale. Return corrected values.

#         Parameters
#         ----------
#             time_obj : numpy.ndarray[float]
#                 List to be corrected to the Barycentric Julian Date in TDB time scale.
#             obj_location : numpy.ndarray[float]
#                 List of the RA and DEC in degrees of the object being observed.
#             obs_location : Optional(numpy.ndarray[float])                           NOTE considered optional only if keyword argument - check for these
#                 List of the longitude and latitude in degrees of the site of observation. If None given, uses gravitational center of Earth at North Pole.
       
#         Returns
#         -------
#             time_obj.value : numpy.ndarray[float]
#                 Returned only if no correction needed. Placeholder array of 1s with same shape as `mid_transit_times`.
#             corrected_time_vals : numpy.ndarray[float]
#                 List now corrected to Barycentric Julian Date in TDB time scale.
#         """
#         # If given uncertainties, check they are actual values and not placeholders vals of 1
#         # If they are placeholder vals, no correction needed, just return array of 1s
#         if np.all(time_obj.value == 1):
#             return time_obj.value
#         time_obj.location = obs_location
#         ltt_bary = time_obj.light_travel_time(obj_location)
#         corrected_time_vals = (time_obj.tdb+ltt_bary).value
#         return corrected_time_vals
    
#     def _validate_times(self, mid_transit_times_obj, mid_transit_times_uncertainties_obj, obj_coords, obs_coords):
#         """Checks that object and observatory coordinates are in correct format for correction function, passes the observed transit midpoints and the uncertainties in observed transit midpoints to the correction function. 
    
#         STEP 1: Checks if there are object coordinates (right ascension and declination). Raises a ValueError if not.

#         STEP 2: Checks if there are observatory coordinates (latitude and longitude). Raises a warning and uses the gravitational center of Earth at the North Pole if not.

#         STEP 3: Performs corrections to 'mid_transit_times_obj' and 'mid_transit_times_uncertainties_obj' by calling '_calc_barycentric_time'

#         Parameters
#         ----------
#             mid_transit_times_obj : (astropy.time.Time[array, string, string])         
#                 List of transit midpoints, time_format, and time_scale 
#             mid_trnasit_times_uncertainties_obj : Optional(astropy.time.Time[array, string, string])      NOTE also not really uncertainties in strings - maybe reformat the description
#                 List of uncertainties corresponding with transit midpoints, time_format, and time_scale. If given None initailly, have been replaced with array of 1's with same shape as `mid_transit_times`.
#             obj_coords : numpy.ndarray[float]
#                 List of the RA and DEC in degrees of the object being observed.
#             obs_coords : Optional(numpy.ndarray[float])             
#                 List of the longitude and latitude in degrees of the site of observation. If None given, use gravitational center of Earth at North Pole.

#         Raises
#             ValueError :
#                 Error if None recieved for object_ra or object_dec.
#         ------

#         """
#         # check if there are objects coords, raise error if not
#         if all(elem is None for elem in obj_coords):
#             raise ValueError("Recieved None for object right ascension and/or declination. " 
#                              "Please enter ICRS coordinate values in degrees for object_ra and object_dec for TransitTime object.")
#         # Check if there are observatory coords, raise warning and use earth grav center coords if not
#         if all(elem is None for elem in obs_coords):
#             logging.warning(f"Unable to process observatory coordinates {obs_coords}. "
#                              "Using gravitational center of Earth.")
#             obs_location = coord.EarthLocation.from_geocentric(0., 0., 0., unit=u.m)
#         else:
#             obs_location = coord.EarthLocation.from_geodetic(obs_coords[0], obs_coords[1])
#         obj_location = coord.SkyCoord(ra=obj_coords[0], dec=obj_coords[1], unit='deg', frame='icrs')
#         logging.warning(f"Using ICRS coordinates in degrees of RA and Dec {round(obj_location.ra.value, 2), round(obj_location.dec.value, 2)} for time correction. "
#                         f"Using geodetic Earth coordinates in degrees of longitude and latitude {round(obs_location.lon.value, 2), round(obs_location.lat.value, 2)} for time correction.")
#         # Perform correction, will return array of corrected times
#         self.mid_transit_times_uncertainties = self._calc_barycentric_time(mid_transit_times_uncertainties_obj, obj_location, obs_location)
#         self.mid_transit_times = self._calc_barycentric_time(mid_transit_times_obj, obj_location, obs_location)

#     def _validate(self):
#         """Checks that all object attributes are of correct types and within value constraints.
#         STEP 1: Check all object attributes are of type array.

#         STEP 2: Check all object attributes are of same shape.

#         STEP 3: Check all object attributes contain correct value type.

#         STEP 4: Check all object attributes contain no null values.

#         STEP 5: Check 'mid_transit_times_uncertainties' contains non-negative and non-zero values.

#         Raises
#         ------
#             TypeError :
#                 Error if 'epochs', 'mid_traisit_times', or 'mid_transit_times_uncertainties' are not NumPy arrays.
#             ValueError :
#                 Error if shapes of 'epochs', 'mid_transit_times', and 'mid_transit_times_uncertainties' arrays do not match.
#             TypeError :
#                 Error if values in 'epochs' are not ints, values in 'mid_transit_times' or 'mid_transit_times_uncertainties" are not floats. 
#             ValueError :
#                 Error if 'epochs', 'mid_transit_times', or 'mid_transit_times_uncertainties' contain a NaN (Not-a-Number) value.
#             ValueError :
#                 Error if 'mid_transit_times_uncertainties' contains a negative or zero value.
#         """
#         # Check that all are of type array
#         if not isinstance(self.epochs, np.ndarray):
#             raise TypeError("The variable 'epochs' expected a NumPy array (np.ndarray) but received a different data type")
#         if not isinstance(self.mid_transit_times, np.ndarray):
#             raise TypeError("The variable 'mid_transit_times' expected a NumPy array (np.ndarray) but received a different data type")
#         if not isinstance(self.mid_transit_times_uncertainties, np.ndarray):
#             raise TypeError("The variable 'mid_transit_times_uncertainties' expected a NumPy array (np.ndarray) but received a different data type")
#         # Check that all are same shape
#         if self.epochs.shape != self.mid_transit_times.shape != self.mid_transit_times_uncertainties.shape:
#             raise ValueError("Shapes of 'epochs', 'mid_transit_times', and 'mid_transit_times_uncertainties' arrays do not match.")
#         # Check that all values in arrays are correct
#         # if not all(isinstance(value, (int, np.int64)) for value in self.epochs) or not all(isinstance(value, (int, np.int32)) for value in self.epochs):
#         if not all(isinstance(value, (int, np.int64, np.int32)) for value in self.epochs):
#             raise TypeError("All values in 'epochs' must be of type int, numpy.int64, or numpy.int32.")
#         if not all(isinstance(value, float) for value in self.mid_transit_times):
#             raise TypeError("All values in 'mid_transit_times' must be of type float.")
#         if not all(isinstance(value, float) for value in self.mid_transit_times_uncertainties):
#             raise TypeError("All values in 'mid_transit_times_uncertainties' must be of type float.")
#         # Check that there are no null values
#         if np.any(np.isnan(self.epochs)):
#             raise ValueError("The 'epochs' array contains NaN (Not-a-Number) values.")
#         if np.any(np.isnan(self.mid_transit_times)):
#             raise ValueError("The 'mid_transit_times' array contains NaN (Not-a-Number) values.")
#         if np.any(np.isnan(self.mid_transit_times_uncertainties)):
#             raise ValueError("The 'mid_transit_times_uncertainties' array contains NaN (Not-a-Number) values.")
#         # Check that mid_transit_times_uncertainties are positive and non-zero (greater than zero)
#         if not np.all(self.mid_transit_times_uncertainties > 0):
#             raise ValueError("The 'mid_transit_times_uncertainties' array must contain non-negative and non-zero values.")
#         if self.epochs[0] != 0:
#             # Shift epochs and mid transit times
#             self.epochs -= np.min(self.epochs)
#             # TODO import warning that we are minimizing their epochs and transit times
#         if self.mid_transit_times[0] != 0:
#             self.mid_transit_times -= np.min(self.mid_transit_times)