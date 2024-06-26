import numpy as np
from astropy import time
from astropy import coordinates as coord
from astropy import units as u
import logging

# logging.basicConfig(format='%(levelname)s: %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
# logger = getLogger(__name__)

logger = logging.getLogger('lumberjack')

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
        # Configure logging to remove 'root' prefix
        self._configure_logging()

        self.epochs = epochs
        self.mid_times = mid_times
        self.tra_or_occ = tra_or_occ
        if mid_time_uncertainties is None:
            # If no uncertainties provided, make an array of 1s in the same shape of epochs
            mid_time_uncertainties =  np.ones_like(self.epochs, dtype=float)
        self.mid_time_uncertainties = mid_time_uncertainties
        # Check that timing system and scale are JD and TDB
        if time_format != 'jd' or time_scale != 'tdb':
            # If not correct time format and scale, create time objects and run corrections
            logging.warning(f"Recieved time format {time_format} and time scale {time_scale}. Correcting all times to BJD timing system with TDB time scale. If no time scale is given, default is automatically assigned to UTC. If this is incorrect, please set the time format and time scale for TransitTime object.")
            # Set timing data to None for now
            self.mid_times = None
            self.mid_time_uncertainties = None
            mid_times_obj = time.Time(mid_times, format=time_format, scale=time_scale)
            mid_time_uncertainties_obj = time.Time(mid_time_uncertainties, format=time_format, scale=time_scale)
            self._validate_times(mid_times_obj, mid_time_uncertainties_obj, (object_ra, object_dec), (observatory_lon, observatory_lat))
        # Call validation function
        self._validate()

    def _configure_logging(self):
        logging.basicConfig(format='%(levelname)s: %(message)s')

    def _calc_barycentric_time(self, time_obj, obj_location, obs_location):
        """Function to correct non-barycentric time formats to Barycentric Julian Date in TDB time scale.

        This function will run under the given circumstances:
            * If the timing format provided is not JD (time_format == 'jd')
            * If the timing scale is not provided
            * If the timing scale provided is not TDB (time_sclae == 'tdb')

        Calculates the light travel time for the given Astropy timing object and adds the light 
        travel time to each original value in the given timing data. If the given Astropy timing object time data 
        contains a list of 1s, which means this is placeholder timing uncertainty data, no timing correction will 
        be applied as this is not real data. If the timing correction proceeds, the `light_travel_time` function 
        from Astropy will be applied and added to the original timing data. Timing data corrected for Barycentric  
        light travel time will be returned.

        Parameters
        ----------
            time_obj : numpy.ndarray[float]
               List of timing data to be corrected to the Barycentric Julian Date time format in the TDB time scale.
            obj_location : Astropy.coordinates.SkyCoord obj
                The RA and DEC in degrees of the object being observed, stored as an Astropy coordinates.SkyCoord object.
            obs_location : Astropy.coordinates.EarthLocation obj
                The longitude and latitude in degrees of the site of observation, stored as an Astropy coordinates.EarthLocation object. 
                If None given, uses gravitational center of Earth at North Pole.
       
        Returns
        -------
            time_obj.value : numpy.ndarray[float]
                Returned only if these are placeholder uncertainties and no correction is needed.
            corrected_time_vals : numpy.ndarray[float]
                List of mid times corrected to Barycentric Julian Date time format with TDB time scale.
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
        """Validates object and observatory information and populates Astropy objects for barycentric light travel time correction.

        Checks that object coordinates (right ascension and declination) and are of correct types. If correct object 
        coordinates are given, will create an Astropy SkyCoord object. Checks that observatory coordinates (longitude 
        and latitude) are given and of correct types. If given, will populate an Astropy EarthLocation object. If not
        given, will populate Astropy EarthLocation with gravitational center of Earth at North Pole. Passes the validated
        SkyCoord, EarthLocation, and Time objects to the `_calc_barycentric_time` correction function to convert times
        to BJD TDB timing format and scale.

        Parameters
        ----------
            mid_times_obj : (astropy.time.Time[numpy.ndarray[floats], string, string])
                An Astropy Time object containing the mid time data, time format, and time scale
            mid_time_uncertainties_obj : Optional(astropy.time.Time[numpy.ndarray[floats], string, string])
                An Astropy Time object containing timing uncertainty data, time format, and time scale. If no uncertainties intially given, data has been replaced with array of 1's with same shape as `mid_times`.
            obj_coords : object_ra: float, object_dec: float)
                Tuple of the right ascension and declination in degrees of the object being observed.
            obs_coords : Optional((observatory_lon: float, observatory_lat: float))    
                Tuple of the longitude and latitude in degrees of the site of observation.

        Raises
            ValueError :
                Error if None recieved for object_ra or object_dec.
            Warning:
                Warning if no observatory coordinates are given.
                Warning notifying user that barycentric light travel time correction is taking place with the given
                information.
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
        logging.warning(f"Using ICRS coordinates in degrees of RA and Dec {round(obj_location.ra.value, 2), round(obj_location.dec.value, 2)} for time correction. Using geodetic Earth coordinates in degrees of longitude and latitude {round(obs_location.lon.value, 2), round(obs_location.lat.value, 2)} for time correction.")
        # Perform correction, will return array of corrected times
        self.mid_time_uncertainties = self._calc_barycentric_time(mid_time_uncertainties_obj, obj_location, obs_location)
        self.mid_times = self._calc_barycentric_time(mid_times_obj, obj_location, obs_location)

    def _validate_tra_or_occ(self):
        """TODO: Write docstring
        
        """
        # Check that object is of type array
        if not isinstance(self.tra_or_occ, np.ndarray):
            raise TypeError("The variable 'tra_or_occ' expected a NumPy array (np.ndarray) but received a different data type")
        # Check if any values are not valid in tra_or_occ array
        if any(val not in ['tra', 'occ'] for val in self.tra_or_occ):
            raise ValueError("tra_or_occ array cannot contain string values other than 'tra' or 'occ'")
        # Check the shape 
        if self.tra_or_occ.shape != self.mid_time_uncertainties.shape != self.mid_times.shape != self.epochs.shape:
            raise ValueError("Shapes of 'tra_or_occ', 'mid_time_uncertainties', 'mid_times', and 'epochs' arrays do not match.")
        # null values
        if np.issubdtype(self.tra_or_occ.dtype, np.number) and np.any(np.isnan(self.tra_or_occ)):
            raise ValueError("The 'tra_or_occ' array contains NaN (Not-a-Number) values.")
   
    def _validate(self):
        """Checks that all object attributes are of correct types and within value constraints.

        Validates the types of the main data attributes: epochs, mid_times, and mid_time_uncertainties. 
        The following checks are in place:
            * Contained in numpy arrays
            * Epochs are integers
            * Mid times and uncertainties are floats
            * No null or nan values are in the data
            * Epochs, mid times, and uncertainties all contain the same amount of data points
            * All uncertainties are positive
        If the epochs and mid times do not start at zero, the arrays will be shifted to start at zero by 
        subtracting the minimum value of the array from each data point in the array.

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
        # if self.mid_times[0] != 0:
        #     self.mid_times -= np.min(self.mid_times)
        if self.tra_or_occ is None:
            # Create list of just 'tra'
            self.tra_or_occ = np.array(['tra' for element in self.epochs])
        if self.tra_or_occ is not None:
            self._validate_tra_or_occ()