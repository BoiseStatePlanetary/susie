import numpy as np
from astropy import time
from astropy import coordinates as coord
from astropy import units as u
import logging

logger = logging.getLogger("lumberjack")

class TimingData():
    """Represents timing mid-point data over observations. Holds data to be accessed by :class:`Ephemeris`.

    This object processes, formats, and holds user data to be passed to the :class:`Ephemeris` object.
    Users can input observational data as lists of transit and/or occultation mid-times and corresponding
    epochs and (if available) mid-time uncertainties. 
    
    Timing conversions are applied to ensure that all data is processed correctly and users are aware of 
    timing formats and scales, which can give rise to false calculations in our metrics. If data is not specified
    to be in Barycentric Julian Date format with the TDB time scale, timing data will be corrected for barycentric
    light travel time using the Astropy Time utilities. If correction is needed, users will be required to provide
    additional information on the observed object.
    
    If mid time uncertainties are not provided, we will generate placeholder values of 1.

    Our implementations rely on Numpy functions. This object implements checks to ensure that data are stored in 
    Numpy arrays and are of correct data types. The appropriate Type or Value Error is raised if there are any 
    issues.

    All timing data arrays will be sorted by ascending epoch. Epochs are shifted to start at zero by subtracting 
    the minimum number from each value.

    Parameters
    ------------
        time_format: str 
            An abbreviation of the data's time system. Abbreviations for systems can be found on [Astropy's 
            Time documentation](https://docs.astropy.org/en/stable/time/#id3).
        epochs: numpy.ndarray[int]
            List of orbit number reference points for timing observations
        mid_times: numpy.ndarray[float]
            List of observed timing mid points corresponding with epochs, in timing units given by time_format.
        mid_time_uncertainties: Optional(numpy.ndarray[float])
            List of uncertainties corresponding to timing mid points, in timing units given by time_format. If 
            given None, will be replaced with array of 1's with same shape as `mid_times`.
        tra_or_occ: Optional(numpy.ndarray[str])
            List of either `tra` or `occ` which specifies if observational data was taken from a transit or an 
            occultation.
        time_scale: Optional(str)
            An abbreviation of the data's time scale. Abbreviations for scales can be found on [Astropy's Time 
            documentation](https://docs.astropy.org/en/stable/time/#id6).
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
        TypeError: 
            - If ``epochs``, ``mid_times``, ``mid_time_uncertainties``, and/or ``tra_or_occs`` are not Numpy arrays.
            - If ``epochs`` contain any non-int values.
            - If ``mid_times`` and/or ``mid_time_uncertainties`` contain any non-float values.

        ValueError:
            - If ``epochs``, ``mid_times``, ``mid_time_uncertainties``, and/or ``tra_or_occs`` do not have the same 
            amount of data (the arrays do not have the same shape).
            - If ``tra_or_occ`` contains any values that are not 'tra' or 'occ'.
            - If ``mid_times`` or ``mid_time_uncertainties`` contain any NaN values.
            - If ``mid_time_uncertainties`` contain any negative or zero values.
    """
    def __init__(self, time_format, epochs, mid_times, mid_time_uncertainties=None, tra_or_occ=None, time_scale=None, object_ra=None, object_dec=None, observatory_lon=None, observatory_lat=None):
        # Configure logging to remove "root" prefix
        self._configure_logging()
        self.epochs = epochs
        self.mid_times = mid_times
        self.mid_time_uncertainties = mid_time_uncertainties
        if tra_or_occ is None:
            # Create list of just "tra"
            tra_or_occ = np.array(["tra" for el in self.epochs])
        self.tra_or_occ = tra_or_occ
        # Check that timing system and scale are JD and TDB
        if time_format != "jd" or time_scale != "tdb":
            # If not correct time format and scale, create time objects and run corrections
            logging.warning(f"Recieved time format {time_format} and time scale {time_scale}. Correcting all times to BJD timing system with TDB time scale. If no time scale is given, default is automatically assigned to UTC. If this is incorrect, please set the time format and time scale for TransitTime object.")
            # Set timing data to None for now
            self.mid_times = self._convert_times(mid_times, time_format, time_scale, (object_ra, object_dec), (observatory_lon, observatory_lat), warn=True)
            if mid_time_uncertainties is not None:
                self.mid_time_uncertainties = self._convert_timing_uncertainties(mid_times, mid_time_uncertainties, time_format, time_scale, (object_ra, object_dec), (observatory_lon, observatory_lat))
        if mid_time_uncertainties is None:
            # If no uncertainties provided, make an array of 1s in the same shape of epochs
            self.mid_time_uncertainties = np.full(self.epochs.shape, 0.001)
            # self.mid_time_uncertainties = np.ones_like(self.epochs, dtype=float)
        # Call validation function
        self._validate()
        # Once every array is populated, make sure you sort by ascending epoch
        self._sort_data_arrays()

    def _calc_barycentric_time(self, time_obj, obj_location):
        """Function to correct non-barycentric time formats to Barycentric Julian Date in TDB time scale.

        This function will run under the given circumstances:

        - If the timing format provided is not JD (``time_format`` does not equal "jd")
        - If the timing scale provided is not TDB (``time_scale`` does not equal "tdb")
        - If the timing scale is not provided

        Calculates the light travel time for the given Astropy timing object and adds the light 
        travel time to each original value in the given timing data. If the given Astropy timing object time data 
        contains a list of 1s, which means this is placeholder timing uncertainty data, no timing correction will 
        be applied as this is not real data. If the timing correction proceeds, the ``light_travel_time`` function 
        from Astropy will be applied and added to the original timing data. Timing data corrected for Barycentric  
        light travel time will be returned.

        Parameters
        ----------
            time_obj : numpy.ndarray[float]
               List of timing data to be corrected to the Barycentric Julian Date time format in the TDB time scale.
            obj_location : Astropy.coordinates.SkyCoord obj
                The RA and Dec in degrees of the object being observed, stored as an Astropy coordinates.SkyCoord object.
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
        ltt_bary = time_obj.light_travel_time(obj_location)
        corrected_time_vals = (time_obj.tdb+ltt_bary).value
        return corrected_time_vals
    
    def _configure_logging(self):
        logging.basicConfig(format="%(levelname)s: %(message)s")

    def _convert_timing_uncertainties(self, mid_times, mid_time_uncertainties, format, scale, obj_coords, obs_coords):
        """Calculates the converted mid-time uncertainties.

        Calculates the new converted timing uncertainties by calculating the upper and lower limits
        of each mid-time, converting the time formats and scales of the upper and lower limits, then
        subtracting the converted mid time from the limits and taking the average for the final
        resulting mid-time uncertainty.

        This function will ONLY run if the timing format and/or scale needs to be converted and mid-time
        uncertainties are given. If no mid-time uncertainties are given, this calculation will not run and
        placeholder uncertainty values will be generated instead.

        Parameters
        ----------
            mid_times: numpy.ndarray[float]
                List of observed timing mid-points corresponding with epochs.
            mid_time_uncertainties: numpy.ndarray[float]
                List of uncertainties corresponding to timing mid-points.
            format: str
                A valid Astropy abbreviation of the data's time system.
            scale: str
                A valid Astropy abbreviation of the data's time scale.
            obj_coords: (float, float)
                Tuple of the right ascension and declination in degrees of the object being observed.
            obs_coords: (float, float)    
                Tuple of the longitude and latitude in degrees of the site of observation.

        Returns
        -------
            An array of timing uncertainty data converted to Barycentric Julian Date timing format and scale (Astropy JD format, TDB scale).
        """
        # create time objects from upper and lower vals
        mid_times_obj = time.Time(mid_times, format=format, scale=scale)
        upper_times_obj = time.Time(mid_times + mid_time_uncertainties, format=format, scale=scale)
        lower_times_obj = time.Time(mid_times - mid_time_uncertainties, format=format, scale=scale)
        # convert the format mid_times, up and down errs
        mid_times_converted = self._convert_times(mid_times_obj, format, scale, obj_coords, obs_coords)
        upper_times_converted = self._convert_times(upper_times_obj, format, scale, obj_coords, obs_coords)
        lower_times_converted = self._convert_times(lower_times_obj, format, scale, obj_coords, obs_coords)
        # subtract up and down errs
        upper_diffs = upper_times_converted - mid_times_converted
        lower_diffs = mid_times_converted - lower_times_converted
        avg_diff = (upper_diffs + lower_diffs) / 2
        return avg_diff

    def _convert_times(self, time_data, format, scale, obj_coords, obs_coords, warn=False):
        """Validates object and observatory information and populates Astropy objects for barycentric light travel time correction.

        Checks that object coordinates (right ascension and declination) and are of correct types. If correct object 
        coordinates are given, will create an Astropy SkyCoord object. Checks that observatory coordinates (longitude 
        and latitude) are given and of correct types. If given, will populate an Astropy EarthLocation object. If not
        given, will populate Astropy EarthLocation with gravitational center of Earth at North Pole. Passes the validated
        SkyCoord, EarthLocation, and Time objects to the `_calc_barycentric_time` correction function to convert times
        to BJD TDB timing format and scale.

        Parameters
        ----------
            time_data: np.ndarray[float]
                An array of timing data values. This will either be mid times or the mid time uncertainties.
            format: str
                A valid Astropy abbreviation of the data's time system.
            scale: str
                A valid Astropy abbreviation of the data's time scale.
            obj_coords: (float, float)
                Tuple of the right ascension and declination in degrees of the object being observed.
            obs_coords: (float, float)    
                Tuple of the longitude and latitude in degrees of the site of observation.
            warn: Boolean
                If True, will raise warnings to the user.

        Returns
        -------
            An array of timing data converted to Barycentric Julian Date timing format and scale (Astropy JD format, TDB scale).

        Raises
        ------
            ValueError:
                Error if None recieved for object_ra or object_dec.
            Warning:
                Warning if no observatory coordinates are given.
                Warning notifying user that barycentric light travel time correction is taking place with the given
                information.
        """
        # Get observatory and object location
        obs_location = self._get_obs_location(obs_coords, warn)
        obj_location = self._get_obj_location(obj_coords)
        if warn:
            logging.warning(f"Using ICRS coordinates in degrees of RA and Dec {round(obj_location.ra.value, 2), round(obj_location.dec.value, 2)} for time correction. Using geodetic Earth coordinates in degrees of longitude and latitude {round(obs_location.lon.value, 2), round(obs_location.lat.value, 2)} for time correction.")
        # Create time object and convert format to JD
        time_obj = time.Time(time_data, format=format, scale=scale, location=obs_location)
        time_obj_converted_format = time.Time(time_obj.to_value("jd"), format="jd", scale=scale, location=obs_location)
        # Perform barycentric correction for scale conversion, will return array of corrected times
        return self._calc_barycentric_time(time_obj_converted_format, obj_location)
    
    def _get_obj_location(self, obj_coords):
        """Creates the Astropy SkyCoord object for BJD time conversions.

        Parameters
        ----------
            obj_coords: (float, float)
                Tuple of the right ascension and declination in degrees of the object being observed.
        
        Returns
        -------
            An Astropy SkyCoord object with the right ascension and declination in degrees.

        Raises
        ------
            ValueError is there is no data for right ascension and/or declination.
        """
        # check if there are objects coords, raise error if not
        if all(elem is None for elem in obj_coords):
            raise ValueError("Recieved None for object right ascension and/or declination. " 
                             "Please enter ICRS coordinate values in degrees for object_ra and object_dec for TransitTime object.")
        # Create SkyCoord object
        return coord.SkyCoord(ra=obj_coords[0], dec=obj_coords[1], unit="deg", frame="icrs")
    
    def _get_obs_location(self, obs_coords, warn):
        """Creates the EarthLocation object for the BJD time conversion.

        Parameters
        ----------
            obs_coords: (float, float)
                Tuple of the longitude and latitude in degrees of the site of observation.
            warn: Boolean
                If True, will raise warnings to the user.
        
        Returns
        -------
            An Astropy EarthLocation object with the latitude and longitude in degrees.
        """
        # Check if there are observatory coords, raise warning and use earth grav center coords if not
        if all(elem is None for elem in obs_coords):
            if warn:
                logging.warning(f"Unable to process observatory coordinates {obs_coords}. "
                                "Using gravitational center of Earth.")
            return coord.EarthLocation.from_geocentric(0., 0., 0., unit=u.m)
        else:
            return coord.EarthLocation.from_geodetic(obs_coords[0], obs_coords[1])
        
    def _sort_data_arrays(self):
        """Sorts all data arrays by ascending epoch.

        Reorders the epochs, mid_times, mid_time_uncertainties, and tra_or_occ arrays based
        on the index order of ascending epochs. This makes sure that all data is order from first 
        observation to most recent observation.
        """
        sorted_indices = np.argsort(self.epochs)
        self.epochs = self.epochs[sorted_indices]
        self.tra_or_occ = self.tra_or_occ[sorted_indices]
        self.mid_times = self.mid_times[sorted_indices]
        self.mid_time_uncertainties = self.mid_time_uncertainties[sorted_indices]
   
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
                Error if "epochs", "mid_traisit_times", or "mid_time_uncertainties" are not NumPy arrays.
            ValueError :
                Error if shapes of "epochs", "mid_times", and "mid_time_uncertainties" arrays do not match.
            TypeError :
                Error if values in "epochs" are not ints, values in "mid_times" or "mid_time_uncertainties" are not floats. 
            ValueError :
                Error if "epochs", "mid_times", or "mid_time_uncertainties" contain a NaN (Not-a-Number) value.
            ValueError :
                Error if "mid_time_uncertainties" contains a negative or zero value.
        """
        # Check that all are of type array
        if not isinstance(self.epochs, np.ndarray):
            raise TypeError("The variable `epochs` expected a NumPy array (np.ndarray) but received a different data type")
        if not isinstance(self.mid_times, np.ndarray):
            raise TypeError("The variable `mid_times` expected a NumPy array (np.ndarray) but received a different data type")
        if not isinstance(self.mid_time_uncertainties, np.ndarray):
            raise TypeError("The variable `mid_time_uncertainties` expected a NumPy array (np.ndarray) but received a different data type")
        if not isinstance(self.tra_or_occ, np.ndarray):
            raise TypeError("The variable `tra_or_occ` expected a NumPy array (np.ndarray) but received a different data type")
        # Check that all are same shape
        if self.epochs.shape != self.mid_times.shape != self.mid_time_uncertainties.shape != self.tra_or_occ.shape:
            raise ValueError("Shapes of `epochs`, `mid_times`, `mid_time_uncertainties`, and `tra_or_occ` arrays do not match.")
        # Check that all values in arrays are correct
        if not all(isinstance(value, (int, np.int64, np.int32)) for value in self.epochs):
            raise TypeError("All values in `epochs` must be of type int, numpy.int64, or numpy.int32.")
        if not all(isinstance(value, float) for value in self.mid_times):
            raise TypeError("All values in `mid_times` must be of type float.")
        if not all(isinstance(value, float) for value in self.mid_time_uncertainties):
            raise TypeError("All values in `mid_time_uncertainties` must be of type float.")
        if any(val not in ["tra", "occ"] for val in self.tra_or_occ):
            raise ValueError("The `tra_or_occ` array cannot contain string values other than `tra` or `occ`")
        # Check that there are no null values
        if np.any(np.isnan(self.mid_times)):
            raise ValueError("The `mid_times` array contains NaN (Not-a-Number) values.")
        if np.any(np.isnan(self.mid_time_uncertainties)):
            raise ValueError("The `mid_time_uncertainties` array contains NaN (Not-a-Number) values.")
        # Check that mid_time_uncertainties are positive and non-zero (greater than zero)
        if not np.all(self.mid_time_uncertainties > 0):
            raise ValueError("The `mid_time_uncertainties` array must contain non-negative and non-zero values.")
        # Shift epochs by subtracting the minimum number from everything (new list will start at 0)
        if self.epochs[0] != 0:
            self.epochs -= np.min(self.epochs)
            # TODO import warning that we are minimizing their epochs