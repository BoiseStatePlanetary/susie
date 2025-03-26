import pytz
import numpy as np
import pandas as pd
from astropy.units import Quantity
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget, Observer, EclipsingSystem, AtNightConstraint, AltitudeConstraint, is_event_observable
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

class Campaign():
    def __init__(self):
        pass

    def _query_nasa_exoplanet_archive(self, obj_name, ra=None, dec=None, select_query=None):
        """Queries the NASA Exoplanet Archive for system parameters.

        Parameters
        ----------
            obj_name: str
                The name of the exoplanet object.
            ra: float (Optional)
                The right ascension of the object to observe in the sky (most likely a planet or star).
            dec: float (Optional)
                The declination of the object to observe in the sky (most likely a planet or star).
            select_query: str
                The select query string. For examples please see the Astroquery documentation.
            
        Returns
        -------
            obj_data: An astropy Table object
                The table of data returned from the NASA Exoplanet Archive query. For more information on 
                how to work with these tables, please see the Astroquery NASA Exoplanet Archive documentation.
        """
        # Get object data
        obj_data = None
        if obj_name is not None:
            if select_query:
                obj_data = NasaExoplanetArchive.query_object(obj_name, select=select_query)
            else:
                obj_data = NasaExoplanetArchive.query_object(obj_name)
        elif ra is not None and dec is not None:
            if select_query:
                obj_data = NasaExoplanetArchive.query_region(
                    table="pscomppars", coordinates=SkyCoord(ra=ra*u.deg, dec=dec*u.deg),
                    radius=1.0*u.deg, select=select_query)
            else:
                obj_data = NasaExoplanetArchive.query_region(
                    table="pscomppars", coordinates=SkyCoord(ra=ra*u.deg, dec=dec*u.deg), radius=1.0*u.deg)
        else:
            raise ValueError("Object must be specified with either (1) a recognized object name in the NASA Exoplanet Archive or (2) right ascension and declination in degrees as accepted by astropy.coordinates.ra and astropy.coordinates.dec.")
        # Check that the returned data is not empty
        if obj_data is not None and len(obj_data) > 0:
            return obj_data
        else:
            if obj_name is not None:
                raise ValueError(f"Nothing found for {obj_name} in the NASA Exoplanet Archive. Please check that your object is accepted and contains data on the archive's homepage.")
            elif ra is not None and dec is not None:
                raise ValueError(f"Nothing found for the coordinates {ra}, {dec} in the NASA Exoplanet Archive. Please check that your values are correct and are in degrees as accepted by astropy.coordinates.ra and astropy.coordinates.dec.")
    
    def _get_eclipse_duration(self, obj_name, ra=None, dec=None):
        """Queries the NASA Exoplanet Archive for system parameters used in eclipse duration calculation.

        Parameters
        ----------
            obj_name: str
                The name of the exoplanet object.
            ra: float (Optional)
                The right ascension of the object to observe in the sky (most likely a planet or star).
            dec: float (Optional)
                The declination of the object to observe in the sky (most likely a planet or star).
            
        Returns
        -------
        """
        nea_data = self._query_nasa_exoplanet_archive(obj_name, ra, dec, select_query="pl_trandur")
        for val in nea_data["pl_trandur"]:
            if not(np.isnan(val)):
                val_to_store = val
                if isinstance(val, Quantity) and hasattr(val, 'mask'):
                    # If the value is masked, just store value
                    val_to_store = val.value
                return val_to_store * u.hour
            
    def _validate_obs_start_time(self, time_str):
        value_err_msg = "obs_start_time must be in the format YYYY-MM-DD. For example, 2024-10-29."
        if len(time_str) != 10:
            return ValueError(value_err_msg)
        if not (time_str[0:4].isdigit() and time_str[5:7].isdigit() and time_str[8:10].isdigit()):
            return ValueError(value_err_msg)
        if not (time_str[4] == '-' and time_str[7] == '-'):
            return ValueError(value_err_msg)
        
    def _validate_observing_schedule_params(self, observer, target, obs_start_time):
        self._validate_obs_start_time(obs_start_time)
        if not isinstance(observer, Observer):
            return TypeError("observer parameter must be an Astroplan Observer object. See the Astroplan documentation for more information: https://astroplan.readthedocs.io/en/latest/api/astroplan.Observer.html")
        if not isinstance(target, FixedTarget):
            return TypeError("target parameter must be an Astroplan FixedTarget object. See the Astroplan documentation for more information: https://astroplan.readthedocs.io/en/latest/api/astroplan.Target.html")
        
    def _create_observing_schedule_dataframe(self, transits, occultations, time_format):
        transit_df = pd.DataFrame(transits)
        occultation_df = pd.DataFrame(occultations)
        if time_format == "datetime":
            transit_df = transit_df.map(lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S %z"))
            occultation_df = occultation_df.map(lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S %z"))
        transit_df = transit_df.rename(columns={0: "ingress", 1: "egress"})
        occultation_df = occultation_df.rename(columns={0: "ingress", 1: "egress"})
        transit_df["type"] = "transit"
        occultation_df["type"] = "occultation"
        final_df = pd.concat([transit_df, occultation_df], ignore_index=True)
        sorted_df = final_df.sort_values(by="ingress", ascending=True)
        return sorted_df

    def create_observer_obj(self, timezone, name, longitude=None, latitude=None, elevation=0.0):
        """Creates the Astroplan Observer object.

        Parameters
        ----------
            timezone: str
                The local timezone. If a string, it will be passed through pytz.timezone() to produce the timezone object.
            name: str
                The name of the observer's location. This can either be a registered Astropy site
                name (get the latest site names with `EarthLocation.get_site_names()`), which will
                return the latitude, longitude, and elevation of the site OR it can be a custom name
                to keep track of your Observer object.
            latitude: float (Optional)
                The latitude of the observer's location on Earth.
            longitude: float (Optional)
                The longitude of the observer's location on Earth.
            elevation: float (Optional)
                The elevation of the observer's location on Earth.

        Returns
        -------
            The Astroplan Observer object.
        
        Raises
        ------
            ValueError if neither coords nor name are given.
        """
        observer = None
        if longitude is not None and latitude is not None:
            # There are valid vals for lon and lat
            observer = Observer(longitude=longitude*u.deg, latitude=latitude*u.deg, elevation=elevation*u.m, timezone=timezone)
            if name is not None:
                observer.name = name
        elif name is not None:
            # No vals for lon and lat, use site name
            observer = Observer.at_site(name, timezone=timezone)
        else:
            # No coords or site name given, raise error
            raise ValueError("Observatory location must be specified with either (1) a site name specified by astropy.coordinates.EarthLocation.get_site_names() or (2) latitude and longitude in degrees as accepted by astropy.coordinates.Latitude and astropy.coordinates.Latitude.")
        return observer
            
    def create_target_obj(self, name, ra=None, dec=None):
        """Creates the Astroplan FixedTarget object.

        Parameters
        ----------
            coords: tuple(float, float) (Optional)
                The right ascension and declination of the object in the sky (most likely the planet's host star).
            name: str
                The name of the exoplanet host star. This can either be a registered object name, which will query
                a CDS name resolver (see the `Astroplan Target Documentation <https://astroplan.readthedocs.io/en/latest/api/astroplan.Target.html>`_ 
                for more information on this) OR it can be a custom name to keep track of your FixedTarget object.
            ra: float (Optional)
                The right ascension of the object to observe in the sky (most likely a planet or star).
            dec: float (Optional)
                The declination of the object to observe in the sky (most likely a planet or star).
        
        Returns
        -------
            The Astroplan FixedTarget object.

        Raises
        ------
            ValueError if neither coords nor name are given.
        """
        target = None
        if ra is not None and dec is not None:
            # There are valid vals for ra and dec
            skycoord = SkyCoord(ra=ra*u.deg, dec=dec*u.deg)
            target = FixedTarget(coord=skycoord)
            if name is not None:
                target.name = name
        elif name is not None:
            # No vals for ra & dec, query by object name
            target = FixedTarget.from_name(name)
        else:
            # Neither ra & dec or name given, raise error
            raise ValueError("Object location must be specified with either (1) an valid object name or (2) right ascension and declination in degrees as accepted by astropy.coordinates.ra and astropy.coordinates.dec.")
        return target
    
    def get_observing_schedule(self, primary_eclipse_time, orbital_period, timezone, observer, target, n_transits, n_occultations, obs_start_time, exoplanet_name=None, eclipse_duration=None, csv_filename=None, return_time_format="datetime"):
        """Returns a list of observable future transits for the target object

        Parameters
        ----------
            primary_eclipse_time: Astropy Time obj
                The most recent mid-transit time with Julian Date time format.
            orbital_period: float
                The orbital period of the exoplanet in days.
            timezone: str
                The local timezone. If a string, it will be passed through `pytz.timezone()` to produce the timezone object.
            observer: Astroplan Observer obj
                An Astroplan Observer object holding information on the observer's Earth location. Can be created 
                through the `create_observer_obj` method, or can be manually created. See the `Astroplan Observer Documentation <https://astroplan.readthedocs.io/en/latest/api/astroplan.Observer.html>`_
                for more information.
            target: Astroplan FixedTarget obj
                An Astroplan FixedTarget object holding information on the object observed. Can be created through the 
                `create_target_obj` method, or can be manually created. See the `Astroplan Target Documentation <https://astroplan.readthedocs.io/en/latest/api/astroplan.Target.html>`_
                for more information.
            n_transits: int
                The number of transits to initially request. This will be filtered down by what is observable from
                the Earth location.
            n_occultations: int
                The number of occultations to initially request. This will be filtered down by what is observable from
                the Earth location.
            obs_start_time: str
                Time at which you would like to start looking for eclipse events. In the format YYYY-MM-DD. For
                example, if you would like to find eclipses happening after October 1st, 2024, the format would
                be "2024-10-01".
            exoplanet_name: str (Optional)
                The name of the exoplanet. Used to query the NASA Exoplanet Archive for transit duration. If 
                no name is provided, the right ascension and declination from the FixedTarget object will be used. 
                Can also provide the transit duration manually instead using the `eclipse_duration` parameter.
            eclipse_duration: float (Optional)
                The full duration of the exoplanet transit from ingress to egress in hours. If not given, will calculate
                using either provided system parameters or parameters pulled from the NASA Exoplanet Archive.
            csv_filename: str (Optional)
                If given, will save the returned schedule dataframe as a CSV file.
            csv_time_format: str (Optional)
                The time format given in the returned CSV file. Default of datetime format "%Y-%m-%d %H:%M:%S %z".
        """
        # Validate some things before continuing
        self._validate_observing_schedule_params(observer, target, obs_start_time)
        # Grab the most recent mid transit time
        primary_eclipse_time = Time(primary_eclipse_time, format='jd')
        # Pull orbital period from the model data
        orbital_period = orbital_period * u.day
        if eclipse_duration == None:
            # If not given, query the archive for it
            eclipse_duration = self._get_eclipse_duration(exoplanet_name, target.ra, target.dec)
        # Create EclipsingSystem object
        eclipsing_system = EclipsingSystem(primary_eclipse_time=primary_eclipse_time,
                                orbital_period=orbital_period, duration=eclipse_duration)
        # Set the observational parameters
        # Time to start looking
        obs_time = Time(f"{obs_start_time} 00:00")
        # Grab the number of transits and occultations asked for
        ing_egr_transits = eclipsing_system.next_primary_ingress_egress_time(obs_time, n_eclipses=n_transits)
        ing_egr_occultations = eclipsing_system.next_secondary_ingress_egress_time(obs_time, n_eclipses=n_occultations)
        # We need to check if the events are observable
        constraints = [AtNightConstraint.twilight_civil(), AltitudeConstraint(min=30*u.deg)]
        transits_bool = is_event_observable(constraints, observer, target, times_ingress_egress=ing_egr_transits)
        occultations_bool = is_event_observable(constraints, observer, target, times_ingress_egress=ing_egr_occultations)
        observable_transits = ing_egr_transits[transits_bool[0]]
        observable_occultations = ing_egr_occultations[occultations_bool[0]]
        # Change to their given timezone
        if return_time_format == "datetime":
            tz = pytz.timezone(timezone)
            observable_transits = observable_transits.to_datetime(timezone=tz)
            observable_occultations = observable_occultations.to_datetime(timezone=tz)
        # Create dataframe from this
        schedule_df = self._create_observing_schedule_dataframe(observable_transits, observable_occultations, return_time_format)
        if csv_filename is not None:
            schedule_df.to_csv(csv_filename, index=False)
        return schedule_df