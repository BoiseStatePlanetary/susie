from abc import ABC, abstractmethod
import pytz
import pyvo
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lmfit import Model
from astropy.units import Quantity
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget, Observer, EclipsingSystem, AtNightConstraint, AltitudeConstraint, is_event_observable
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
from susie.timing_data import TimingData # Use this for package pushes
# from .timing_data import TimingData # Use this for running tests
# from timing_data import TimingData # Use this for running this file

class BaseModel(ABC):
    """Abstract class that defines the structure of different model classes."""
    @abstractmethod
    def get_initial_params(self, x, y, tra_or_occ):
        """Defines the structure for retrieving initial parameters for the model fit method.
        
        Parameters
        ----------
        x : numpy.ndarray[int]
            The epoch data as received from the TimingData object.
        y : numpy.ndarray[float]
            The mid-time data as received from the TimingData object.
        tra_or_occ: numpy.ndarray[str]
            An array indicating the type of each event, with entries being either 
            "tra" for transit or "occ" for occultation.

        Returns
        -------
        A dictionary containing initial parameter values for the LMfit Model fitting procedure.
        """
        pass

    @abstractmethod
    def fit(self, x, y, yerr, tra_or_occ):
        """Fits a model to timing data.

        Defines the structure for fitting an model (linear, quadratic or precession) to timing data. 
        All subclasses must implement this method.

        Parameters
        ----------
            x : numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            y : numpy.ndarray[float]
                The mid-time data as recieved from the TimingData object.
            yerr : numpy.ndarray[float]
                The mid-time error data as recieved from the TimingData object.
            tra_or_occ: numpy.ndarray[str]
                An array indicating the type of each event, with entries being either 
                "tra" for transit or "occ" for occultation.

        Returns
        ------- 
            A dictionary containing model best-fit parameter values. 
        """
        pass


class LinearModel(BaseModel):
    """Subclass of BaseModel that implements a linear fit."""
    def get_initial_params(self, x, y, tra_or_occ):
        """Computes and returns the initial parameters for the model fit method.

        This method calculates the conjunction time (T0) and orbital period (P) 
        using the provided timing data. The conjunction time is set to the first 
        transit time, while the orbital period is estimated as the median of the 
        differences between consecutive mid-times of transits and occultations.

        Parameters
        ----------
        x : numpy.ndarray[int]
            The epoch data as recieved from the TimingData object.
        y : numpy.ndarray[float]
            The mid-time data as received from the TimingData object.
        tra_or_occ : numpy.ndarray[str]
            An array indicating the type of each event, with entries being either 
            "tra" for transit or "occ" for occultation.

        Returns
        -------
        dict
            A dictionary containing:
            - "conjunction_time" (float): The mid-time of the transit events corresponding to Epoch = 0.
            - "period" (float): The median time difference between consecutive events 
            (transits and occultations).
        """
        tra_mask = tra_or_occ == "tra"
        occ_mask = tra_or_occ == "occ"
        # Getting difference between the first and last values of tra array
        mid_time_diff_tra = abs(np.roll(y[tra_mask], -1)-y[tra_mask])
        epochs_diff_tra = abs(np.roll(x[tra_mask], -1)-x[tra_mask])
        # Dividing to get the period and accessing final value if it exists
        period_tra = np.divide(mid_time_diff_tra, epochs_diff_tra)[-1] if x[tra_mask].size > 0 else np.nan
        # Getting difference between the first and last values of occ array
        mid_time_diff_occ = abs(np.roll(y[occ_mask], -1)-y[occ_mask])
        epochs_diff_occ = abs(np.roll(x[occ_mask], -1)-x[occ_mask])
        # Dividing to get the period and accessing final value if it exists
        period_occ = np.divide(mid_time_diff_occ, epochs_diff_occ)[-1] if x[occ_mask].size > 0 else np.nan
        # Finding final period by getting average of the two
        period = np.nanmean([period_tra, period_occ])
        # Conjunction time (we assume to use transits, use occultations if there are no transits)
        T0 = y[np.where(x == 0) if len(np.where(x == 0)[0]) > 0 else 0]
        return_data = {
            "conjunction_time": T0,
            "period": period
        }
        return return_data
    
    def lin_fit(self, E, T0, P, tra_or_occ_enum):
        """Calculates a linear function with given data.

        Uses the equation 
         - (period * epochs + initial mid time) for transit observations 
         - (period * epochs + (initial mid time + (½ * period ))) for occultation observations 
        
        as a linear function for the LMfit Model.
        
        Parameters
        ----------
            E: numpy.ndarray[float]
                The epoch data as recieved from the TimingData object.
            T0: float
                The initial mid-time, also known as conjunction time.
            P: float
                The exoplanet orbital period.
            tra_or_occ_enum: numpy.ndarray[int]
                An array indicating the type of each event, with enumerated entries being either 
                0 for transit or 1 for occultation.
        
        Returns
        -------
            result: numpy.ndarray[float]
                A linear function to be used with the LMfit Model, returned as:
                    :math:`P*E + T_0` if the data point is an observed transit (denoted by 0)
                    :math:`P*E + (T_0 + \\frac{1}{2}*P)` if the data point is an observed occultation (denoted by 1)
        """
        result = np.zeros(len(E))
        tra_mask = tra_or_occ_enum == 0
        occ_mask = tra_or_occ_enum == 1
        result[tra_mask] = T0 + P*E[tra_mask]
        result[occ_mask] = (T0 + 0.5*P) + P*E[occ_mask]
        return result
    
    def fit(self, x, y, yerr, tra_or_occ):
        """Fits observed data to a linear model.

        Compares the model data from the TimingData object to the linear fit calculated with 
        lin_fit method. Then minimizes the difference between the two sets of data. The LMfit Model then 
        returns the parameters of the linear function corresponding to period, conjunction time, and their 
        respective errors. These parameters are returned in a dictionary to the user.

        Parameters
        ----------
            x: numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            y: numpy.ndarray[float]
                The mid-time data as recieved from the TimingData object.
            yerr: numpy.ndarray[float]
                The mid-time error data as recieved from the TimingData object.
            tra_or_occ: numpy.ndarray[str]
                An array indicating the type of each event, with entries being either 
                "tra" for transit or "occ" for occultation.

        Returns
        ------- 
        return_data: dict
            A dictionary of best-fit parameter values from the fit model.
            Example:

            .. code-block:: python

                {
                'period': Estimated orbital period of the exoplanet (in units of days),
                'period_err': Uncertainty associated with orbital period (in units of days),
                'conjunction_time': Time of conjunction of exoplanet transit or occultation,
                'conjunction_time_err': Uncertainty associated with conjunction_time
                }
        """
        tra_or_occ_enum = [0 if i == 'tra' else 1 for i in tra_or_occ]
        model = Model(self.lin_fit, independent_vars=["E", "tra_or_occ_enum"])
        init_params = self.get_initial_params(x, y, tra_or_occ)
        params = model.make_params(T0=init_params["conjunction_time"], P=init_params["period"], tra_or_occ_enum=tra_or_occ_enum)
        result = model.fit(y, params, weights=1.0/yerr, E=x, tra_or_occ_enum=tra_or_occ_enum)
        return_data = {
            "period": result.params["P"].value,
            "period_err": result.params["P"].stderr,
            "conjunction_time": result.params["T0"].value,
            "conjunction_time_err": result.params["T0"].stderr
        }
        return(return_data)


class QuadraticModel(BaseModel):
    """Subclass of BaseModel that implements a quadratic fit."""
    def get_initial_params(self, x, y, tra_or_occ):
        """Computes and returns the initial parameters for the model fit method.

        This method calculates the conjunction time (T0) and orbital period (P) 
        using the provided timing data. The conjunction time is set to the first 
        transit time, while the orbital period is estimated as the median of the 
        differences between consecutive mid-times of transits and occultations.

        Parameters
        ----------
        x : numpy.ndarray[int]
            The epoch data as recieved from the TimingData object.
        y : numpy.ndarray[float]
            The mid-time data as received from the TimingData object.
        tra_or_occ : numpy.ndarray[str]
            An array indicating the type of each event, with entries being either 
            "tra" for transit or "occ" for occultation.

        Returns
        -------
        dict
            A dictionary containing:
            - "conjunction_time" (float): The mid-time of the transit events corresponding to Epoch = 0.
            - "period" (float): The median time difference between consecutive events 
            (transits and occultations).
        """
        tra_mask = tra_or_occ == "tra"
        occ_mask = tra_or_occ == "occ"
        # Getting difference between the first and last values of tra array
        mid_time_diff_tra = abs(np.roll(y[tra_mask], -1)-y[tra_mask])
        epochs_diff_tra = abs(np.roll(x[tra_mask], -1)-x[tra_mask])
        # Dividing to get the period and accessing final value if it exists
        period_tra = np.divide(mid_time_diff_tra, epochs_diff_tra)[-1] if x[tra_mask].size > 0 else np.nan
        # Getting difference between the first and last values of occ array
        mid_time_diff_occ = abs(np.roll(y[occ_mask], -1)-y[occ_mask])
        epochs_diff_occ = abs(np.roll(x[occ_mask], -1)-x[occ_mask])
        # Dividing to get the period and accessing final value if it exists
        period_occ = np.divide(mid_time_diff_occ, epochs_diff_occ)[-1] if x[occ_mask].size > 0 else np.nan
        # Finding final period by getting average of the two
        period = np.nanmean([period_tra, period_occ])
        # Conjunction time (we assume to use transits, use occultations if there are no transits)
        T0 = y[np.where(x == 0) if len(np.where(x == 0)[0]) > 0 else 0]
        return_data = {
            "conjunction_time": T0,
            "period": period
        }
        return return_data

    def quad_fit(self, E, T0, P, dPdE, tra_or_occ_enum):
        """Calculates a quadratic function with given data.

        Uses the equation 
         - ((0.5 * change in period over epoch * (epoch²)) + (period * epoch) + conjunction time) for transit observations
         - ((0.5 * change in period over epoch * (epoch²)) + (period * epoch) + conjunction time) for occultation observations as a quadratic function for the LMfit Model.
        
        Parameters
        ----------
            E: numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            T0: float
                The initial mid-time, also known as conjunction time.
            P: float
                The exoplanet orbital period.
            dPdE: float
                Change in period with respect to epoch.
            tra_or_occ_enum: numpy.ndarray[int]
                An array indicating the type of each event, with enumerated entries being either 
                0 for transit or 1 for occultation.
        
        Returns
        -------
            result: numpy.ndarray[float]
                A quadratic function to be used with the LMfit Model, returned as:
                    :math:`\\frac{1}{2}*\\frac{dP}{dE}*E^2 + P*E + T_0` if the data point is an observed transit (denoted by 0)
                    :math:`\\frac{1}{2}*\\frac{dP}{dE}*E^2 + P*E + (T_0 + \\frac{1}{2}*P)` if the data point is an observed occultation (denoted by 1)
        """
        result = np.zeros(len(E))
        tra_mask = tra_or_occ_enum == 0
        occ_mask = tra_or_occ_enum == 1
        result[tra_mask] = T0 + P*E[tra_mask] + 0.5*dPdE*np.power(E[tra_mask], 2)
        result[occ_mask] = (T0 + 0.5*P) + P*E[occ_mask] + 0.5*dPdE*np.power(E[occ_mask], 2)
        return result
    
    def fit(self, x, y, yerr, tra_or_occ):
        """Fits observed data to a quadratic model.

        Compares the observed data from the TimingData object to the quadratic fit calculated with quad_fit 
        method. Then minimizes the difference between the two sets of data. The LMfit Model then returns the 
        parameters of the quadratic function corresponding to period, conjunction time, period change by epoch, 
        and their respective errors. These parameters are returned in a dictionary to the user.

        Parameters
        ----------
            x: numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            y: numpy.ndarray[float]
                The mid-time data as recieved from the TimingData object.
            yerr: numpy.ndarray[float]
                The mid-time error data as recieved from the TimingData object.
            tra_or_occ: numpy.ndarray[str]
                An array indicating the type of each event, with entries being either 
                "tra" for transit or "occ" for occultation.

        Returns
        ------- 
        return_data: dict
            A dictionary of best-fit parameter values from the fit model. 
            Example:
                {
                 'period': Estimated orbital period of the exoplanet (in units of days),
                 'period_err': Uncertainty associated with orbital period (in units of days),
                 'conjunction_time': Time of conjunction of exoplanet transit or occultation,
                 'conjunction_time_err': Uncertainty associated with conjunction_time
                 'period_change_by_epoch': The exoplanet period change with respect to epoch (in units of days),
                 'period_change_by_epoch_err': The uncertainties associated with period_change_by_epoch (in units of days)
                }
        """
        tra_or_occ_enum = [0 if i == 'tra' else 1 for i in tra_or_occ]
        model = Model(self.quad_fit, independent_vars=["E", "tra_or_occ_enum"])
        init_params = self.get_initial_params(x, y, tra_or_occ)
        params = model.make_params(T0=init_params["conjunction_time"], P=init_params["period"], dPdE=0.0, tra_or_occ_enum=tra_or_occ_enum)
        result = model.fit(y, params, weights=1.0/yerr, E=x, tra_or_occ_enum=tra_or_occ_enum)
        return_data = {
            "period": result.params["P"].value,
            "period_err": result.params["P"].stderr,
            "conjunction_time": result.params["T0"].value,
            "conjunction_time_err": result.params["T0"].stderr,
            "period_change_by_epoch": result.params["dPdE"].value,
            "period_change_by_epoch_err": result.params["dPdE"].stderr
        }
        return(return_data)
    

class PrecessionModel(BaseModel):
    """ Subclass of BaseModel that implements a precession fit."""
    def get_initial_params(self, x, y, yerr, tra_or_occ):
        """Computes and returns the initial parameters for the model fit method.

        This method calculates the conjunction time (T0) and orbital period (P) 
        using the provided timing data. The conjunction time is set to the first 
        transit time, while the orbital period is estimated as the median of the 
        differences between consecutive mid-times of transits and occultations.

        Parameters
        ----------
        x : numpy.ndarray[int]
            The epoch data as recieved from the TimingData object.
        y : numpy.ndarray[float]
            The mid-time data as received from the TimingData object.
        tra_or_occ : numpy.ndarray[str]
            An array indicating the type of each event, with entries being either 
            "tra" for transit or "occ" for occultation.

        Returns
        -------
        dict
            A dictionary containing:
            - "conjunction_time" (float): The mid-time of the transit events corresponding to Epoch = 0.
            - "period" (float): The median time difference between consecutive events 
            (transits and occultations).
        """
        tra_mask = tra_or_occ == "tra"
        occ_mask = tra_or_occ == "occ"
        # Getting difference between the first and last values of tra array
        mid_time_diff_tra = abs(np.roll(y[tra_mask], -1)-y[tra_mask])
        epochs_diff_tra = abs(np.roll(x[tra_mask], -1)-x[tra_mask])
        # Dividing to get the period and accessing final value if it exists
        period_tra = np.divide(mid_time_diff_tra, epochs_diff_tra)[-1] if x[tra_mask].size > 0 else np.nan
        # Getting difference between the first and last values of occ array
        mid_time_diff_occ = abs(np.roll(y[occ_mask], -1)-y[occ_mask])
        epochs_diff_occ = abs(np.roll(x[occ_mask], -1)-x[occ_mask])
        # Dividing to get the period and accessing final value if it exists
        period_occ = np.divide(mid_time_diff_occ, epochs_diff_occ)[-1] if x[occ_mask].size > 0 else np.nan
        # Finding final period by getting average of the two
        period = np.nanmean([period_tra, period_occ])
        # Conjunction time (we assume to use transits, use occultations if there are no transits)
        T0 = y[np.where(x == 0) if len(np.where(x == 0)[0]) > 0 else 0]
        # T0 = y[tra_mask][0]
        # # Pull the eccentricity from the NASA Exoplanet Archive
        # service = pyvo.dal.TAPService("https://exoplanetarchive.ipac.caltech.edu/TAP")
        # query = f"SELECT pl_orbeccen FROM ps WHERE pl_name='TrES-5 b'"
        # results = service.search(query)
        # table = results.to_table()
        # eccentricity = next((x for x in np.array(table["pl_orbeccen"]) if not np.isnan(x) and x != 0.0), 0.001)
        # # Default of 2 radians (114.592 degrees) if nothing is found
        # # arg_of_periastron = (next((x for x in np.array(table["pl_orblper"]) if not np.isnan(x) and x != 0.0), 114.592) * np.pi) / 180
        # # Calculate the initial values using the quadratic model
        # quad_ephem = Model(TimingData("jd", x, y, yerr, tra_or_occ, "tdb")).fit_model("quadratic")
        # a = 0.5*quad_ephem["period_change_by_epoch"]
        # b = quad_ephem["period"]
        # c = quad_ephem["conjunction_time"]
        # p = [a, b, c]
        # roots = np.roots(p)
        # w_0 = ((((2*a)/(b)) * roots[0] + 0.5) / (((2*a)/(b)) * roots[0] - 1)) * np.pi
        # dwdE = (np.pi) / (roots[0] - roots[1])
        # e = (c - ((b**2)/(4*a))) * ((2*np.pi) / b)
        # return_data = {
        #     "conjunction_time": T0,
        #     "period": period,
        #     "eccentricity": eccentricity,
        #     "pericenter": w_0,
        #     "precession_rate": dwdE
        # }
        return_data = {
            "conjunction_time": T0,
            "period": period
        }
        return return_data
    
    def _anomalistic_period(self, P, dwdE):
       """Calculates the anomalistic period given values of sidereal period (P_s) and precession rate (dwdE).

       Uses the equation:
       P / (1 - (1/(2*pi)) * dwdE)

       Parameters
       ----------
        P: float
           The exoplanet sideral orbital period.
        dwdE: float
           The precession rate, which is the change in pericenter with respect to epoch.

        Returns
        -------
           A float of the calculated anomalistic period.
       """
       result = P/(1 - ((1/(2*np.pi))*dwdE))
       return result
    
    def _pericenter(self, E, w0, dwdE):
       """Calculates the pericenter given a list of epochs and values of argument of pericenter and precession rate.

       Uses the equation:
        w0 + dwdE * E

       Parameters
       ----------
        E: numpy.ndarray[int]
            The epoch data as recieved from the TimingData object.
        w0: int
            The argument of pericenter.
        dwdE: float
            The precession rate, which is the change in pericenter with respect to epoch.
        
        Returns
        -------
           A numpy.ndarray[float] of the calculated pericenter as a function of epochs.
       """
       result = w0 + (dwdE*E)
       return result
    
    def precession_fit(self, E, T0, P, e, w0, dwdE, tra_or_occ_enum):
        """Calculates a precession function with given data.

        Uses the equation 
         -  conjunction time + (epochs * period) - ((eccentricity * anomalistic period) / pi) * cos(pericenter) for transit observations
         -  conjunction time + (anomalistic period / 2) + epochs * period + ((eccentricity * anomalistic period) / pi) * cos(pericenter) for occultation observations as a precession function for the LMfit Model.
        
        Parameters
        ----------
            E: numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            T0: float
                The initial mid-time, also known as conjunction time.
            P: float
                The exoplanet sideral orbital period.
            e: float
                The eccentricity.
            w0: int
                The argument of pericenter.
            dwdE: float
                Precession rate, which is change in pericenter with respect to epoch.
            tra_or_occ_enum: numpy.ndarray[int]
                An array indicating the type of each event, with enumerated entries being either 
                0 for transit or 1 for occultation.
        
        Returns
        -------
            result: numpy.ndarray[float]
                A precession function to be used with the LMfit Model, returned as:
                :math:`T0 + E*P - \\frac{e * \\text{self.anomalistic_period}(P,dwdE)}{\\pi} * \\cos(\\text{self.pericenter}(w0, dwdE, E))`
                :math:`T0 + \\frac{\\text{self.anomalistic_period}(P,dwdE)}{2} + E*P + \\frac{e * \\text{self.anomalistic_period}(P,dwdE)}{\\pi} * \\cos(\\text{self.pericenter}(w0, dwdE, E))`
        """
        result = np.zeros(len(E))
        P_a = self._anomalistic_period(P, dwdE)
        tra_mask = tra_or_occ_enum == 0
        occ_mask = tra_or_occ_enum == 1
        result[tra_mask] = T0 + (E[tra_mask]*P) - ((e*P_a)/np.pi)*np.cos(self._pericenter(E[tra_mask], w0, dwdE))
        result[occ_mask] = T0 + P_a/2 + (E[occ_mask]*P) + ((e*P_a)/np.pi)*np.cos(self._pericenter(E[occ_mask], w0, dwdE))
        return result

    def fit(self, x, y, yerr, tra_or_occ):
        """Fits observed data to a precession model.

        Compares the observed data from the TimingData object to the precession fit calculated with 
        precession_fit method. Then minimizes the difference between the two sets of data. The LMfit Model 
        then returns the parameters of the precession function corresponding to period, conjunction time, 
        pericenter change by epoch, eccentricity, pericenter, and their respective errors. These parameters 
        are returned in a dictionary to the user.

        Parameters
        ----------
            x: numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            y: numpy.ndarray[float]
                The mid-time data as recieved from the TimingData object.
            yerr: numpy.ndarray[float]
                The mid-time error data as recieved from the TimingData object.
            tra_or_occ: numpy.ndarray[str]
                An array indicating the type of each event, with entries being either 
                "tra" for transit or "occ" for occultation.

        Returns
        ------- 
        return_data: dict
            A dictionary of best-fit parameter values from the fit model.
            Example:
                {
                 'period': Estimated orbital period of the exoplanet (in units of days),
                 'period_err': Uncertainty associated with orbital period (in units of days),
                 'conjunction_time': Time of conjunction of exoplanet transit or occultation,
                 'conjunction_time_err': Uncertainty associated with conjunction time,
                 'eccentricity': The eccentricity of the exoplanet's orbit,
                 'eccentricity_err': The uncertainties associated with eccentricity,
                 'pericenter': The argument of pericenter of the exoplanet,
                 'pericenter_err': The uncertainties associated with pericenter,
                 'pericenter_change_by_epoch': The precession rate of the exoplanet's orbit,
                 'pericenter_change_by_epoch_err': The uncertainties associated with precession rate
                }
        """
        # STARTING VAL OF dwdE CANNOT BE 0, WILL RESULT IN NAN VALUES FOR THE MODEL
        tra_or_occ_enum = [0 if i == 'tra' else 1 for i in tra_or_occ]
        model = Model(self.precession_fit, independent_vars=["E", "tra_or_occ_enum"])
        init_params = self.get_initial_params(x, y, yerr, tra_or_occ)
        # TODO: Can put bounds between 0 and 2pi for omega0
        # TODO: Try out bound for d omega dE for abs(dwdE) < 2pi/delta E
        # params = model.make_params(T0=init_params["conjunction_time"], P=init_params["period"], e=dict(value=init_params["eccentricity"], min=0, max=1), w0=dict(value=init_params["pericenter"], min=0, max=2*np.pi), dwdE=dict(value=init_params["precession_rate"]), tra_or_occ_enum=tra_or_occ_enum)
        params = model.make_params(T0=init_params["conjunction_time"], P=init_params["period"], e=dict(value=0.001, min=0, max=1), w0=2.0, dwdE=dict(value=0.001), tra_or_occ_enum=tra_or_occ_enum)
        result = model.fit(y, params, weights=1.0/yerr, E=x, tra_or_occ_enum=tra_or_occ_enum)
        return_data = {
            "period": result.params["P"].value,
            "period_err": result.params["P"].stderr,
            "conjunction_time": result.params["T0"].value,
            "conjunction_time_err": result.params["T0"].stderr,
            "eccentricity": result.params["e"].value,
            "eccentricity_err": result.params["e"].stderr,
            "pericenter": result.params["w0"].value,
            "pericenter_err": result.params["w0"].stderr,
            "pericenter_change_by_epoch": result.params["dwdE"].value,
            "pericenter_change_by_epoch_err": result.params["dwdE"].stderr
        }
        return(return_data)


class ModelFactory:
    """Factory class for selecting which type of model class (linear, quadratic or precession) to use."""
    @staticmethod
    def create_model(model_type, x, y, yerr, tra_or_occ):
        """Instantiates the appropriate BaseModel subclass and runs fit_model method.

        Based on the given user input of model type (linear, quadratic or precession) the factory will 
        create the corresponding subclass of BaseModel and run the fit_model method to recieve the 
        model return data dictionary.
        
        Parameters
        ----------
            model_type: str
                The name of the model to create, either 'linear', 'quadratic' or 'precession'.
            x: numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            y: numpy.ndarray[float]
                The mid-time data as recieved from the TimingData object.
            yerr: numpy.ndarray[float]
                The mid-time error data as recieved from the TimingData object.
            tra_or_occ: numpy.ndarray[str]
                An array indicating the type of each event, with entries being either 
                "tra" for transit or "occ" for occultation.

        Returns
        ------- 
            Model : dict
                A dictionary of parameters from the fit model. If a linear model was chosen, these parameters are:
                    * 'period': Estimated orbital period of the exoplanet (in units of days),
                    * 'period_err': Uncertainty associated with orbital period (in units of days),
                    * 'conjunction_time': Time of conjunction of exoplanet transit or occultation,
                    * 'conjunction_time_err': Uncertainty associated with conjunction_time
                If a quadratic model was chosen, the same variables are returned, and an additional parameter is included in the dictionary:
                    * 'period_change_by_epoch': The exoplanet period change with respect to epoch (in units of days),
                    * 'period_change_by_epoch_err': The uncertainties associated with period_change_by_epoch (in units of days)
                If a precession model was chosen, the same varibales as linear are returned, and additional parameter is included in the dictionary:
                    * 'eccentricity': The eccentricity of the exoplanet's orbit,
                    * 'eccentricity_err': The uncertainties associated with eccentricity,
                    * 'pericenter': The argument of pericenter of the exoplanet,
                    * 'pericenter_err': The uncertainties associated with pericenter,
                    * 'pericenter_change_by_epoch': The precession rate of the exoplanet's orbit,
                    * 'pericenter_change_by_epoch_err': The uncertainties associated with precession rate
        
        Raises
        ------
            ValueError:
                If model specified is not a valid subclass of BaseModel, which is either 'linear', 'quadratic', or 'precession'.
        """
        models = {
            'linear': LinearModel(),
            'quadratic': QuadraticModel(),
            'precession': PrecessionModel()
        }
        if model_type not in models:
            raise ValueError(f"Invalid model type: {model_type}")
        model = models[model_type]
        return model.fit(x, y, yerr, tra_or_occ)


class Ephemeris(object):
    """Represents the fit model using transit or occultation mid-time data over epochs.

    Parameters
    -----------
    timing_data: TimingData obj
        A successfully instantiated TimingData object holding epochs, mid-times, and uncertainties.
        
    Raises
    ----------
     ValueError:
        Raised if timing_data is not an instance of the TimingData object.
    """
    def __init__(self, timing_data):
        """Initializing the model object

        Parameters
        -----------
        timing_data: TimingData obj
            A successfully instantiated TimingData object holding epochs, mid-times, and uncertainties.
        
        Raises
        ------
            ValueError :
                error raised if 'timing_data' is not an instance of 'TimingData' object.
        """
        self.timing_data = timing_data
        self._validate()
        self.tra_mask = self.timing_data.tra_or_occ == "tra"
        self.occ_mask = self.timing_data.tra_or_occ == "occ"

    def _validate(self):
        """Check that timing_data is an instance of the TimingData object.

        Raises
        ------
            ValueError :
                error raised if 'timing_data' is not an instance of 'TimingData' object.
        """
        if not isinstance(self.timing_data, TimingData):
            raise ValueError("Variable 'timing_data' expected type of object 'TimingData'.")
        
    def _get_timing_data(self):
        """Returns timing data for use.

        Returns the epoch, mid-time, and mid-time uncertainty data from the TimingData object.

        Returns
        -------
            x: numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            y: numpy.ndarray[float]
                The mid-time data as recieved from the TimingData object.
            yerr: numpy.ndarray[float]
                The mid-time error data as recieved from the TimingData object.
            tra_or_occ: numpy.ndarray[str]
                An array indicating the type of each event, with entries being either 
                "tra" for transit or "occ" for occultation.
        """
        x = self.timing_data.epochs
        y = self.timing_data.mid_times
        yerr = self.timing_data.mid_time_uncertainties
        tra_or_occ = self.timing_data.tra_or_occ
        return x, y, yerr, tra_or_occ
    
    def _get_model_parameters(self, model_type, **kwargs):
        """Creates the model object and returns model best-fit parameters.
        
        This method fetches data from the TimingData object to be used in the model fit. 
        It creates the appropriate subclass of BaseModel using the Model factory, then runs 
        the fit method to return the model best-fit parameters dictionary to the user.

        Parameters
        ----------
            model_type: str
                Either 'linear', 'quadratic', or 'precession'. The model subclass specified to create and run.

        Returns
        -------
            model_data: dict
                A dictionary of best-fit parameters from the fit model. 
                If a linear model was chosen, these parameters are:
                {
                    'period': Estimated orbital period of the exoplanet (in units of days),
                    'period_err': Uncertainty associated with orbital period (in units of days),
                    'conjunction_time': Time of conjunction of exoplanet transit or occultation,
                    'conjunction_time_err': Uncertainty associated with conjunction_time
                }
                If a quadratic model was chosen, the same linear variables are returned, and an additional parameter is included in the dictionary:
                {
                    'period_change_by_epoch': The exoplanet period change with respect to epoch (in units of days),
                    'period_change_by_epoch_err': The uncertainties associated with period_change_by_epoch (in units of days)
                }
                If a precession model was chosen, the same linear variables are returned, and the following additional parameters are included in the dictionary:
                {
                    'eccentricity': The eccentricity of the exoplanet's orbit,
                    'eccentricity_err': The uncertainties associated with eccentricity,
                    'pericenter': The argument of pericenter of the exoplanet,
                    'pericenter_err': The uncertainties associated with pericenter,
                    'pericenter_change_by_epoch': The precession rate of the exoplanet's orbit,
                    'pericenter_change_by_epoch_err': The uncertainties associated with precession rate
                }

        Raises
        ------
            ValueError:
                If model specified is not a valid subclass of BaseModel, which is either 'linear', 'quadratic', or 'precession'.
        """
        # Step 1: Get data from transit times obj
        x, y, yerr, tra_or_occ = self._get_timing_data()
        # Step 2: Create the model with the given variables & user inputs. 
        # This will return a dictionary with the best-fit parameters as key value pairs.
        model_data = ModelFactory.create_model(model_type, x, y, yerr, tra_or_occ, **kwargs)
        # Step 3: Return the data dictionary with the best-fit parameters
        return model_data
    
    def _get_k_value(self, model_type):
        """Returns the number of fit parameters for a given model.
        
        Parameters
        ----------
            model_type: str
                Either 'linear', 'quadratic', or 'precession', used to specify how many fit parameters are present in the model.

        Returns
        -------
            An int representing the number of fit parameters for the model. This will be: 
                * 2 for a linear model 
                * 3 for a quadratic model
                * 5 for a precession model

        Raises
        ------
            ValueError
                If the model_type is an unsupported model type. Currently supported model types are 
                'linear', 'quadratic', and 'precession'.
        """
        if model_type == 'linear':
            return 2
        elif model_type == 'quadratic':
            return 3
        elif model_type == 'precession':
            return 5
        else:
            raise ValueError('Only linear, quadratic, and precession ephemerides are supported at this time.')
    
    def _calc_anomalistic_period(self, P, dwdE):
       """Calculates the anomalistic period given values of sidereal period and precession rate (change in pericenter over epoch).

        Uses the equation
        :math:`\\frac{P}{(1 - \\frac{1}{2*\\pi})*frac{dw}{dE}}`
       
        Parameters
        ----------
            P: float
                The exoplanet sidereal orbital period.
            dwdE: float
                The precession rate, which is the change in pericenter with respect to epoch.

        Returns
        -------
            A float of the calculated anomalistic period.
       """
       result = P/(1 - (1/(2*np.pi))*dwdE)
       return result
    
    def _calc_pericenter(self, E, w0, dwdE):
       """Calculates the pericenter given a list of epochs and values of argument of pericenter and precession rate (change in pericenter with respect to epoch).

        Uses the equation
        :math:`w0 + \\frac{dw}{dE} * E`


        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs.
            w0: int
                The pericenter.
            dwdE: float
                Change in pericenter with respect to epoch.

        Returns
        -------
           A numpy.ndarray[float] of the calculated pericenter as a function of epochs.
       """
       result = w0 + dwdE*E
       return result
    
    def _calc_linear_model_uncertainties(self, T0_err, P_err):
        """Calculates the uncertainties of a given linear model when compared to observed data in TimingData.
        
        Uses the equation 
         - .. math::
            \\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * E^2)}
            for transit observations
         - .. math::
            \\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * (\\frac{1}{2} + E)^2)}
            for occultation observations
         - σ(t pred, tra) = √(σ(T0)² + σ(P)² * E²) for transit observations
         - σ(t pred, occ) = √(σ(T0)² + σ(P)² * (½ + E)²) for occultation observations
         
        where σ(T0)=conjunction time error, E=epoch, and σ(P)=period error, to calculate the uncertainties 
        between the model data and observed data.
        
        Parameters
        ----------
            T0_err: float
                The error associated with the best-fit conjunction time from a linear model.
            P_err: float
                The error associated with the best-fit period from a linear model.
        
        Returns
        -------
            A list of uncertainties associated with the model passed in, calculated with the 
            equation above and the TimingData epochs.
        """
        result = np.zeros(len(self.timing_data.epochs))
        result[self.tra_mask] = np.sqrt((T0_err**2) + ((self.timing_data.epochs[self.tra_mask]**2)*(P_err**2)))
        result[self.occ_mask] = np.sqrt((T0_err**2) + (((self.timing_data.epochs[self.occ_mask]+0.5)**2)*(P_err**2)))
        return result
    
    def _calc_quadratic_model_uncertainties(self, T0_err, P_err, dPdE_err):
        """Calculates the uncertainties of a given quadratic model when compared to observed data in TimingData.
        
        Uses the equation 
         - .. math::
            \\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * E^2) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))} 
            for transit observations
         - .. math::
            \\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * (\\frac{1}{2} + E^2)) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))} 
            for occultation observation��(σ(T0)² + (σ(P)² * E²) + (¼ * σ(dP/dE)² * E⁴)) for transit observation
         - σ(t pred, occ) = √(σ(T0)² + (σ(P)² * (½ + E)²) + (¼ * σ(dP/dE)² * E⁴)) for occultation observations
        where σ(T0)=conjunction time error, E=epoch, σ(P)=period error, and σ(dP/dE)=period change with respect 
        to epoch error, to calculate the uncertainties between the model data and the observed data.
        
        Parameters
        ----------
            T0_err: float
                The error associated with the best-fit conjunction time from a quadratic model.
            P_err: float
                The error associated with the best-fit period from a quadratic model.
            dPdE_err: float
                The error associated with the best-fit decay rate from a quadratic model.
        
        Returns
        -------
            A list of uncertainties associated with the model passed in, calculated with the 
            equation above and the TimingData epochs.
        """
        result = np.zeros(len(self.timing_data.epochs))
        result[self.tra_mask] = np.sqrt((T0_err**2) + ((self.timing_data.epochs[self.tra_mask]**2)*(P_err**2)) + ((1/4)*(self.timing_data.epochs[self.tra_mask]**4)*(dPdE_err**2)))
        result[self.occ_mask] = np.sqrt((T0_err**2) + (((self.timing_data.epochs[self.occ_mask]+0.5)**2)*(P_err**2)) + ((1/4)*(self.timing_data.epochs[self.occ_mask]**4)*(dPdE_err**2)))
        return result
    
    # ————————————————————WORK IN PROGRESS—————————————————————————————

    # # Precession Uncertainites
    # def _calc_t0_model_uncertainty(self, T0_err):
    #     return T0_err**2

    # def _calc_eccentricity_model_uncertainty(self, P, dwdE, w0, E, e_err):
    #     return (-P/((1-(1/(2*np.pi))*dwdE)*np.pi) * np.cos(w0 + dwdE*E))**2 * e_err**2

    # def _calc_pericenter_model_uncertainty(self, e, P, dwdE, w0, E, w0_err):
    #     return ((e*P)/((1-(1/(2*np.pi))*dwdE)*np.pi) * np.sin(w0 + dwdE*E))**2 * w0_err**2
    
    # def _calc_change_in_pericenter_transit_model_uncertainty(self, e, P, dwdE, w0, E, dwdE_err):
    #     return (((-2*e*P)/((1-(1/(2*np.pi))*dwdE)**2)) * np.cos(w0 + dwdE*E) + E*np.sin(w0 + dwdE*E) * ((-e*P)/((1-(1/(2*np.pi))*dwdE)*np.pi)))**2 * dwdE_err**2
    
    # def _calc_change_in_pericenter_occ_model_uncertainty(self, e, P, dwdE, w0, E, dwdE_err):
    #     return (((np.pi*P)/((1-(1/(2*np.pi))*dwdE)**2)) + ((2*e*P)/((1-(1/(2*np.pi))*dwdE)**2)) * np.cos(w0 + dwdE*E) + E*np.sin(w0 + dwdE*E) * ((e*P)/((1-(1/(2*np.pi))*dwdE)*np.pi)))**2 * dwdE_err**2

    # def _calc_period_transit_model_uncertainty(self, e, dwdE, w0, E, P_err):
    #     return (E - e/((1-(1/(2*np.pi))*dwdE)*np.pi) * np.cos(w0 + dwdE*E))**2 * P_err**2
        
    # def _calc_period_occ_model_uncertainty(self, e, dwdE, w0, E, P_err):
    #     return (E + e/(2*(1-(1/(2*np.pi))*dwdE)) + e/((1-(1/(2*np.pi))*dwdE)*np.pi)* np.cos(w0 + dwdE*E))**2 * P_err**2

    # # def _get_precession_model_partial_derivatives(self, tra_or_occ, epoch)
    # #     if tra:
    # #         return [self._calc_t0_model_uncertainty(T0), self._calc_eccentricity_model_uncertainty(P, dwdE, w0, epoch, e_err), self._calc_pericenter_model_uncertainty(e, P, dwdE, w0, epoch, w0_err), self._calc_change_in_pericenter_transit_model_uncertainty( e, P, dwdE, w0, epoch, dwdE_err), self._calc_period_transit_model_uncertainty(e, dwdE, w0,  self.timing_data.epochs[i], P_err)]
    
    # def _calc_precession_model_uncertainties(self, model_params):
    #     T0_err = model_params['conjunction_time_err']
    #     P_err = model_params['period_err']
    #     dwdE_err = model_params['pericenter_change_by_epoch_err']
    #     e_err = model_params['eccentricity_err']
    #     w0_err = model_params['pericenter_err']
    #     T0 = model_params['conjunction_time']
    #     P = model_params['period']
    #     dwdE = model_params['pericenter_change_by_epoch']
    #     e = model_params['eccentricity']
    #     w0 = model_params['pericenter']       
    #     result = []
    #     for i, t_type in enumerate(self.timing_data.tra_or_occ):
    #         if t_type == 'tra':
    #             # transit data
    #             result.append(np.sqrt(self._calc_t0_model_uncertainty(T0_err) + self._calc_eccentricity_model_uncertainty(P, dwdE, w0,self.timing_data.epochs[i], e_err) + self._calc_pericenter_model_uncertainty(e, P, dwdE, w0, self.timing_data.epochs[i], w0_err) + self._calc_change_in_pericenter_transit_model_uncertainty( e, P, dwdE, w0, self.timing_data.epochs[i], dwdE_err) + self._calc_period_transit_model_uncertainty(e, dwdE, w0,  self.timing_data.epochs[i], P_err)))
    #         elif t_type == 'occ':
    #             # occultation data
    #             result.append(np.sqrt(self._calc_t0_model_uncertainty(T0_err) + self._calc_eccentricity_model_uncertainty(P, dwdE, w0,self.timing_data.epochs[i], e_err) + self._calc_pericenter_model_uncertainty(e, P, dwdE, w0, self.timing_data.epochs[i], w0_err) + self._calc_change_in_pericenter_occ_model_uncertainty( e, P, dwdE, w0, self.timing_data.epochs[i], dwdE_err) + self._calc_period_occ_model_uncertainty(e, dwdE, w0,  self.timing_data.epochs[i], P_err)))
    #     return np.array(result)

    # ———————————————————————————————————————————————————————————————————————————————————————————
    
    def _calc_linear_model(self, E, T0, P):
        """Calculates mid-times using best-fit parameter values from a linear model fit.
        
        Uses the equation:
         - (T0 + PE) for transit observations
         - ((T0 + ½P) + PE) for occultation observations
        to calculate the mid-times for each epoch where T0 is conjunction time, P is period, 
        and E is epoch.

        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs pulled from the TimingData object.
            T0: float
                The best-fit conjunction time from a linear model fit.
            P: float
                The best-fit orbital period from a linear model fit.

        Returns
        -------
            A numpy array of mid-times calculated for each epoch.
        """
        result = np.zeros(len(self.timing_data.epochs))
        result[self.tra_mask] = T0 + (P*E[self.tra_mask])
        result[self.occ_mask] = (T0 + 0.5*P) + (P*E[self.occ_mask])
        return result
    
    def _calc_quadratic_model(self, E, T0, P, dPdE):
        """Calculates mid-times using best-fit parameter values from a quadratic model fit.

        Uses the equation:
         - (T0 + PE + (½ * dPdE * E²)) for transit observations
         - ((T0 + ½P) + PE + (½ * dPdE * E²)) for occultation observations
        to calculate the mid-times over each epoch where T0 is conjunction time, P is period, E is epoch, 
        and dPdE is the orbital decay rate (period change with respect to epoch).

        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs pulled from the TimingData object.
            T0: float
                The best-fit conjunction time from a quadratic model fit.
            P: float
                The best-fit orbital period from a quadratic model fit.
            dPdE: float
                The best-fit orbital decay rate (period change with respect to epoch) from a quadratic model fit.

        Returns
        -------
            A numpy array of mid-times calculated for each epoch.
        """
        result = np.zeros(len(self.timing_data.epochs))
        result[self.tra_mask] = T0 + P*E[self.tra_mask] + 0.5*dPdE*E[self.tra_mask]*E[self.tra_mask]
        result[self.occ_mask] = (T0 + 0.5*P) + P*E[self.occ_mask] + 0.5*dPdE*E[self.occ_mask]*E[self.occ_mask]
        return result
    
    def _calc_precession_model(self, E, T0, P, e, w0, dwdE):
        """Calculates mid-times using best-fit parameter values from a precession model fit.

        Uses the equation:
         -  T0 + E*P - (e*anomalistic period)/pi * cos(pericenter) for transit observations
         -  T0 + (anomalistic period / 2) + (e*anomalistic period)/pi * cos(pericenter) for occultation observations
        to calculate the mid-times over each epoch where T0 is conjunction time, P is sideral period, E is epoch, 
        dwdE is the precession rate (pericenter change with respect to epoch), w0 is the argument of pericenter, 
        and e is eccentricity.

        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs.
            T0: float
                The initial mid-time, also known as conjunction time.
            P: float
                The sideral orbital period of the exoplanet.
            e: float
                The eccentricity.
            w0: int
                The argument of pericenter.
            dwdE: float
                The precession rate, which is the change in pericenter with respect to epoch.
            tra_or_occ: numpy.ndarray[str]
                An array indicating the type of each event, with entries being either 
                "tra" for transit or "occ" for occultation.

        Returns
        -------
            A numpy array of mid-times calculated for each epoch.
        """ 
        result = np.zeros(len(self.timing_data.epochs))
        result[self.tra_mask] = T0 + E[self.tra_mask]*P - ((e*self._calc_anomalistic_period(P, dwdE))/np.pi)*np.cos(self._calc_pericenter(E[self.tra_mask], w0, dwdE))
        result[self.occ_mask] = T0 + (self._calc_anomalistic_period(P, dwdE)/2) + (E[self.occ_mask]*P) + ((e*self._calc_anomalistic_period(P, dwdE))/np.pi)*np.cos(self._calc_pericenter(E[self.occ_mask], w0, dwdE))
        return result
    
    def _calc_sum_of_errs(self, numerator, denominator):
        """Calculates the sum of minimized errors. 

            Equations are pulled from "Numerical Recipes: The Art of Scientific Computing"
            3rd Edition by Press et. al on page 781 (eq. 15.2.4). 

            Σ (numerator / (denominator)²)

            Parameters
            ----------
                numerator: np.array[float]
                    The numerator value.
                denominator: np.array[float]
                    The denominator value.

            Returns
            -------
                A calculated float value using the equation: Σ (numerator / (denominator)²)
        """
        return np.sum((numerator / np.power(denominator, 2)))
   
    def _calc_chi_squared(self, model_mid_times):
        """Calculates the residual chi squared values comparing the model fit and observed data.
        
        Parameters
        ----------
            model_mid_times : numpy.ndarray[float]
                Mid-times calculated from an model. This data can be accessed through the 'model_data'
                key from a returned model data dictionary. 
        
        Returns
        -------
            Chi-squared value : float
                The chi-squared value calculated with the equation:
                Σ(((observed mid-times - model calculated mid-times) / observed mid-time uncertainties)²)
        """
        # STEP 1: Get observed mid-times
        observed_data = self.timing_data.mid_times
        uncertainties = self.timing_data.mid_time_uncertainties
        # STEP 2: calculate X² with observed data and model data
        return np.sum(((observed_data - model_mid_times)/uncertainties)**2)
    
    def _calc_delta_T0_prime_quad(self):
        """Calculates the value of ΔT0' for a quadratic model, used in the analytical ΔBIC equation.

            Equation is pulled Metrics for Optimizing Searches for Tidally Decaying Exoplanets by 
            Jackson et. al (2023) (eq. 32).

            ΔT0' = (S_(e²)² - S_(e³)*S_e) / (S_(e²)*S - S_e²)
        """
        sigma = self.timing_data.mid_time_uncertainties
        epochs = self.timing_data.epochs
        S = self._calc_sum_of_errs(1, sigma)
        S_e = self._calc_sum_of_errs(epochs, sigma)
        S_e2 = self._calc_sum_of_errs(np.power(epochs, 2), sigma)
        S_e3 = self._calc_sum_of_errs(np.power(epochs, 3), sigma)
        delta_T0_prime = (S_e2**2 - (S_e3*S_e)) / ((S_e2*S) - S_e**2)
        return delta_T0_prime
    
    def _calc_delta_P_prime_quad(self):
        """Calculates the value of ΔP' for a quadratic model, used in the analytical ΔBIC equation.
            
            Equation is pulled Metrics for Optimizing Searches for Tidally Decaying Exoplanets by 
            Jackson et. al (2023) (eq. 33).
            
            ΔP' = (S_(e³)*S - S_(e²)*S_e) / (S_(e²)*S - S_e²)
        """
        sigma = self.timing_data.mid_time_uncertainties
        epochs = self.timing_data.epochs
        S = self._calc_sum_of_errs(1, sigma)
        S_e = self._calc_sum_of_errs(epochs, sigma)
        S_e2 = self._calc_sum_of_errs(np.power(epochs, 2), sigma)
        S_e3 = self._calc_sum_of_errs(np.power(epochs, 3), sigma)
        delta_P_prime = (S_e3*S - S_e2*S_e) / ((S_e2*S) - S_e**2)
        return delta_P_prime
    
    def _calc_analytical_delta_bic_quad(self, dPdE):
        """Calculates the analytical ΔBIC for a quadratic model.

            Equation is pulled Metrics for Optimizing Searches for Tidally Decaying Exoplanets by 
            Jackson et. al (2023) (eq. 35).

            ΔBIC = 0.25(dPdE)² * sum(((E² - ΔP'*E - ΔT0') / (uncertainties))²) - ln(N) + 1
            Where dPdE is the orbital decay rate, E are epochs, uncertainties are the uncertainties on the
            observed mid-times, and N is the total number of data points. Equations for ΔP' and ΔT0' can be 
            found in the methods for _calc_delta_P_prime_quad and _calc_delta_T0_prime_quad.

            Parameters
            ----------
                dPdE: float
                    The best-fit parameter value for the orbital decay rate from a quadratic model fit.

            Returns
            -------
                analytical_delta_bic: np.ndarray[float]
                    The values of the analytical ΔBIC for each epoch.
        """
        epochs = self.timing_data.epochs
        uncertainties = self.timing_data.mid_time_uncertainties
        delta_P_prime = self._calc_delta_P_prime_quad()
        delta_T0_prime = self._calc_delta_T0_prime_quad()
        quad_param = 0.25 * (dPdE**2)
        quad_sum = np.sum(np.power(((np.power(epochs, 2) - (delta_P_prime*epochs) - delta_T0_prime) / (uncertainties)), 2))
        analytical_delta_bic = quad_param * quad_sum - np.log(len(epochs)) + 1
        return analytical_delta_bic

    def _calc_delta_P_prime_prec(self, w0, dwdE):
        """Calculates the value of ΔP' for a precession model, used in the analytical ΔBIC equation.
        
        deltaP_s' = [(S_E * S_cos(omega) - S * S_E,cos(omega)) / (S * S_(E^2) - S_E^2)]

        Parameters
        ----------
            timing_uncertainties: np.ndarray(float)
                The uncertainties of the observed mid-times.
            epochs: np.ndarray(int)
                Epochs of observations
            w_0: float
                The argument of pericenter, or "pericenter" from the precession model
            dwdE: float
                dwdE or "change in pericenter over epochs" from the precession model
                
        Returns
        -------
            delta_P_prime: np.ndarray(float)
        """
        epochs = self.timing_data.epochs
        sigma = self.timing_data.mid_time_uncertainties
        cosw = np.cos(w0 + dwdE * self.timing_data.epochs)
        S = self._calc_sum_of_errs(1, sigma)
        S_e = self._calc_sum_of_errs(epochs, sigma)
        S_e2 = self._calc_sum_of_errs(np.power(epochs, 2), sigma)
        S_cosw = self._calc_sum_of_errs(cosw, sigma)
        S_e_cosw = self._calc_sum_of_errs((epochs*cosw), sigma)
        delta_P_prime = ((S_e * S_cosw) - (S * S_e_cosw)) / ((S * S_e2) - (S_e * S_e))
        return delta_P_prime

    def _calc_delta_T0_prime_prec(self, w0, dwdE):
        """Calculates the value of ΔT0' for a precession model, used in the analytical ΔBIC equation.
        
        deltaT_0' = [(S_E * S_E,cos(omega) - S_(E^2) * S_cos(omega)) / (S * S_(E^2) - S_E^2)]

        Parameters
        ----------
            timing_uncertainties: np.ndarray(float)
                The uncertainties ???
            epochs: np.ndarray(int)
                Epochs of observations
            w_0: float
                Omega_0 or "pericenter" from the precession model
            dwdE: float
                dwdE or "change in pericenter over epochs" from the precession model
                
        Returns
        -------
            delta_T0_prime: np.ndarray(float)
        """
        epochs = self.timing_data.epochs
        sigma = self.timing_data.mid_time_uncertainties
        cosw = np.cos(w0+dwdE*epochs)
        S = self._calc_sum_of_errs(1, sigma)
        S_e = self._calc_sum_of_errs(epochs, sigma)
        S_e2 = self._calc_sum_of_errs(np.power(epochs, 2), sigma)
        S_cosw = self._calc_sum_of_errs(cosw, sigma)
        S_e_cosw = self._calc_sum_of_errs((epochs*cosw), sigma)
        delta_T0_prime = ((S_e * S_e_cosw) - (S_e2 * S_cosw)) / ((S * S_e2) - (S_e * S_e))
        return delta_T0_prime
    
    def _calc_analytical_delta_bic_prec(self, Pa, e, w0, dwdE):
        """Calculates the analytical ΔBIC for a precession model.

        delta BIC = ((e*P_a)/pi)^2 * sum(from i = 0 to N-1) [(timing uncertainties_i)^-2 * (cos(omega_i) + E_i * deltaP_s' + deltaT_0')^2 -3 ln(N) + 3]
        """
        epochs = self.timing_data.epochs
        sigma = self.timing_data.mid_time_uncertainties
        cosw = np.cos(w0 + dwdE*epochs)
        # QUESTION: Does this period need to be recalculated at each step or is this also a constant
        delta_T0_prime = self._calc_delta_T0_prime_prec(w0, dwdE)
        delta_P_prime = self._calc_delta_P_prime_prec(w0, dwdE)
        amplitude = np.power(((e*Pa) / (np.pi)), 2)
        delta_bic_sum = np.sum(np.power(sigma, -2) * np.power((cosw + epochs*delta_P_prime + delta_T0_prime), 2))
        analytical_delta_bic = amplitude * delta_bic_sum - 3.0 * np.log(len(epochs)) + 3.0
        return analytical_delta_bic
    
    def _subtract_linear_parameters(self, model_mid_times, T0, P, E, tra_or_occ):
        """Subtracts the linear terms to show smaller changes in non-linear parameters.

        Uses the equations:
         - (model midtime - T0 - PE) for transit observations
         - (model midtime - T0 - (½P) - PE) for occultation observations
        
        Parameters
        ----------
            model_mid_times : numpy.ndarray[float]
                Mid-times calculated from a model. This data can be accessed through the 'model_data'
                key from a returned model data dictionary. 
            T0: float
                The best-fit conjunction time from an model fit.
            P: float
                The best-fit orbital period from an model fit.
            E: numpy.ndarray[int]
                The epochs pulled from the TimingData object.

        Returns
        -------
            A numpy array of newly calculated values for plotting.
        """
        tra_mask = tra_or_occ == "tra"
        occ_mask = tra_or_occ == "occ"
        result = np.zeros(len(E))
        result[tra_mask] = model_mid_times[tra_mask] - T0 - (P*E[tra_mask])
        result[occ_mask] = model_mid_times[occ_mask] - T0 - (0.5*P) - (P*E[occ_mask])
        return result
    
    def fit_model(self, model_type):
        """Fits the timing data to a specified model using an LMfit Model fit.

        Parameters
        ----------
            model_type: str
                Either 'linear', 'quadratic', or 'precession'. Represents the type of model to fit the data to.

        Returns
        ------- 
            model_data: dict
                A dictionary of best-fit parameters from the model fit.
                    Example:

                    If a linear model is chosen, these parameters are:

                    .. code-block:: python
                        
                        {
                            'model_type': "Either linear, quadratic, or precession",
                            'model_data': "A list of calculated mid-times using the best-fit parameter values for each epoch",
                            'period': "Estimated orbital period of the exoplanet (in units of days)",
                            'period_err': "Uncertainty associated with orbital period (in units of days)",
                            'conjunction_time': "Time of conjunction of exoplanet transit or occultation",
                            'conjunction_time_err': "Uncertainty associated with conjunction_time",
                        }
                    
                    If a quadratic model is chosen, the same linear variables are returned, and the following additional parameters are included in the dictionary:
                    
                    .. code-block:: python
                        
                        {
                            'period_change_by_epoch': "The exoplanet orbital decay rate, which is the change in period with respect to epoch (in units of days)",
                            'period_change_by_epoch_err': "The uncertainties associated with period_change_by_epoch (in units of days)"
                        }

                    If a precession model is chosen, the same linear variables are returned, and the following additional parameters are included in the dictionary:

                    .. code-block:: python

                        {
                            'eccentricity': "The eccentricity of the exoplanet's orbit",
                            'eccentricity_err': "The uncertainties associated with eccentricity",
                            'pericenter': "The argument of pericenter of the exoplanet",
                            'pericenter_err': "The uncertainties associated with pericenter",
                            'pericenter_change_by_epoch': "The precession rate, which is the change in pericenter with respect to epoch, of the exoplanet's orbit",
                            'pericenter_change_by_epoch_err': "The uncertainties associated with precession rate"
                        }
        """
        model_data = self._get_model_parameters(model_type)
        model_data["model_type"] = model_type
        # Once we get parameters back, we call _calc_blank_model 
        if model_type == "linear":
            # Return dict with parameters and calculated mid-times
            model_data["model_data"] = self._calc_linear_model(self.timing_data.epochs, model_data["conjunction_time"], model_data["period"])
        elif model_type == "quadratic":
            model_data["model_data"] = self._calc_quadratic_model(self.timing_data.epochs, model_data["conjunction_time"], model_data["period"], model_data["period_change_by_epoch"])
        elif model_type == "precession":
            model_data["model_data"] = self._calc_precession_model(self.timing_data.epochs, model_data["conjunction_time"], model_data["period"], model_data["eccentricity"], model_data["pericenter"], model_data["pericenter_change_by_epoch"])
        return model_data
    
    def get_model_uncertainties(self, model_data):
        """Calculates the mid-time uncertainties of specific model data when compared to the actual data. 

        Calculate the uncertainties between the model data and actual data over epochs using the equations...
        
        For a linear model:
        
         - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * E^2)}` for transit observations

         - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * (\\frac{1}{2} + E)^2)}` for occultation observations
            
        For a quadratic model:

         - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * E^2) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))}` for transit observations

         - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * (\\frac{1}{2} + E^2)) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))}` for occultation observations
        
        where :math:`\\sigma(T_0) =` conjunction time error, :math:`E=` epoch, :math:`\\sigma(P)=` period error, and :math:`\\sigma(\\frac{dP}{dE})=` period change with respect to epoch error.
        
        Parameters
        ----------
        model_data: dict
            The model data dictionary recieved from the `fit_model` method.
        
        Returns
        -------
            A list of mid-time uncertainties associated with the model passed in, calculated with the 
            equation above and the TimingData epochs.
        
        Raises
        ------
            KeyError
                If the model type in not in the model parameter dictionary.
            KeyError
                If the model parameter error values are not in the model parameter dictionary.
        """
        linear_params = ["conjunction_time", "conjunction_time_err", "period", "period_err"]
        quad_params = ['conjunction_time', 'conjunction_time_err', 'period', 'period_err', 'period_change_by_epoch', 'period_change_by_epoch_err']
        prec_params = ['conjunction_time', 'conjunction_time_err', 'period', 'period_err', 'pericenter_change_by_epoch', 'pericenter_change_by_epoch_err', 'eccentricity', 'eccentricity_err', 'pericenter', 'pericenter_err']
        if 'model_type' not in model_data:
            raise KeyError("Cannot find model type in model data. Please run the fit_model method to return model fit parameters.")
        if model_data['model_type'] == 'linear':
            if not any(np.isin(sorted(linear_params), sorted(model_data.keys()))):
                raise KeyError("Cannot find conjunction time and period and/or their respective errors in model data. Please run the fit_model method with 'linear' model_type to return model fit parameters.")
            return self._calc_linear_model_uncertainties(model_data['conjunction_time_err'], model_data['period_err'])
        elif model_data['model_type'] == 'quadratic':
            if not any(np.isin(sorted(quad_params), sorted(model_data.keys()))):
                raise KeyError("Cannot find conjunction time, period, and/or period change by epoch and/or their respective errors in model data. Please run the fit_model method with 'quadratic' model_type to return model fit parameters.")
            return self._calc_quadratic_model_uncertainties(model_data['conjunction_time_err'], model_data['period_err'], model_data['period_change_by_epoch_err'])
        elif model_data['model_type'] == 'precession':
            if not any(np.isin(sorted(prec_params), sorted(model_data.keys()))):
                raise KeyError("Cannot find conjunction time, period, eccentricity, pericenter, and/or pericenter change by epoch and/or their respective errors in model data. Please run the fit_model method with 'precession' model_type to return model fit parameters.")
            return self._calc_precession_model_uncertainties(model_data)

    def calc_bic(self, model_data):
        """Calculates the BIC value for a given model. 
        
        The BIC value is a modified :math:`\\chi^2` value that penalizes for additional parameters. 
        Uses the equation :math:`BIC = \\chi^2 + (k * log(N))` where :math:`\\chi^2=\\sum{\\frac{(\\text{
        observed midtimes - model midtimes})}{\\text{(observed midtime uncertainties})^2}},`
        k=number of fit parameters (2 for linear ephemerides, 3 for quadratic ephemerides, 5 for precession 
        ephemerides), and N=total number of data points.
        
        Parameters
        ----------
            model_data: dict
                The model data dictionary recieved from the `fit_model` method.
        
        Returns
        ------- 
            A float value representing the BIC value for this model.
        """
        # Step 1: Get value of k based on model_type (linear=2, quad=3, custom=?)
        num_params = self._get_k_value(model_data['model_type'])
        # Step 2: Calculate chi-squared
        chi_squared = self._calc_chi_squared(model_data['model_data'])
        # Step 3: Calculate BIC
        return chi_squared + (num_params*np.log(len(model_data['model_data'])))

    def calc_delta_bic(self, model1="linear", model2="quadratic"):
        """Calculates the :math:`\\Delta BIC` value between linear and quadratic ephemerides using the given timing data. 
        
        model1 BIC value - model2 BIC value, default of linear - quadratic BIC values.
        
        The BIC value is a modified :math:`\\chi^2` value that penalizes for additional parameters. The
        :math:`\\Delta BIC` value is the difference between the linear BIC value and the quadratic BIC value.
        Ephemerides that have smaller values of BIC are favored. Therefore, if the :math:`\\Delta BIC` value for your
        data is a large positive number (large linear BIC - small quadratic BIC), a quadratic model is favored and
        your data indicates possible orbital decay in your extrasolar system. If the :math:`\\Delta BIC` value for
        your data is a small number or negative (small linear BIC - large quadratic BIC), then a linear model is
        favored and your data may not indicate orbital decay. 

        This function will run all model instantiation and BIC calculations for you using the TimingData
        information you entered.

        Parameters
        ----------
            model1: str
                This is the name of the first model.
            model2: str
                This is the name of second model, whose BIC value will be subtracted from the first.

        Returns
        ------- 
            delta_bic : float
                Represents the :math:`\\Delta BIC` value for this timing data. 
        """
        valid_ephemerides = ["linear", "quadratic", "precession"]
        if model1 not in valid_ephemerides or model2 not in valid_ephemerides:
            raise ValueError("Only linear, quadratic, and precession ephemerides are accepted at this time.")
        model1_data = self.fit_model(model1)
        model2_data = self.fit_model(model2)
        model1_bic = self.calc_bic(model1_data)
        model2_bic = self.calc_bic(model2_data)
        delta_bic = model1_bic - model2_bic
        return delta_bic
    
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
        
    def _create_observing_schedule_dataframe(self, transits, occultations):
        transit_df = pd.DataFrame(transits)
        occultation_df = pd.DataFrame(occultations)
        transit_df = transit_df.map(lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S %z"))
        transit_df = transit_df.rename(columns={0: "ingress", 1: "egress"})
        occultation_df = occultation_df.map(lambda dt: dt.strftime("%Y-%m-%d %H:%M:%S %z"))
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
    
    def get_observing_schedule(self, model_data, timezone, observer, target, n_transits, n_occultations, obs_start_time, exoplanet_name=None, eclipse_duration=None, csv_filename=None):
        """Returns a list of observable future transits for the target object

        Parameters
        ----------
            model_data: dict
                The model data dictionary recieved from the `fit_model` method.
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
                The full duration of the exoplanet transit from ingress to egress. If not given, will calculate
                using either provided system parameters or parameters pulled from the NASA Exoplanet Archive.
            csv_filename: str (Optional)
                If given, will save the returned schedule dataframe as a CSV file.
        """
        # Validate some things before continuing
        self._validate_observing_schedule_params(observer, target, obs_start_time)
        # Grab the most recent mid transit time
        primary_eclipse_time = Time(self.timing_data.mid_times[-1], format='jd')
        # Pull orbital period from the model data
        orbital_period = model_data['period'] * u.day
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
        tz = pytz.timezone(timezone)
        converted_transits = observable_transits.to_datetime(timezone=tz)
        converted_occultations = observable_occultations.to_datetime(timezone=tz)
        # Create dataframe from this
        schedule_df = self._create_observing_schedule_dataframe(converted_transits, converted_occultations)
        if csv_filename is not None:
            schedule_df.to_csv(csv_filename, index=False)
        return schedule_df
    
    def plot_model(self, model_data, subtract_lin_params=False, show_occultations=False, save_plot=False, save_filepath=None):
        """Plots a scatterplot of epochs vs. model calculated mid-times.

        Parameters
        ----------
            model_data: dict
                The model data dictionary recieved from the `fit_model` method.
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        DAYS_TO_SECONDS = 1*24*60*60
        fig, ax = plt.subplots(figsize=(6*(16/9), 6))
        y_data = model_data['model_data']
        # Subtract the linear parameters if arg is True
        if subtract_lin_params:
            y_data = self._subtract_linear_parameters(model_data['model_data'], model_data['conjunction_time'], model_data['period'], self.timing_data.epochs, self.timing_data.tra_or_occ)*DAYS_TO_SECONDS
        # Plot transits and occultations separately if arg is True
        if show_occultations:
            ax.plot(self.timing_data.epochs[self.tra_mask], y_data[self.tra_mask], color='#0033A0', label="Transits")
            ax.plot(self.timing_data.epochs[self.occ_mask], y_data[self.occ_mask], color="#D64309", label="Occultations")
            ax.legend()
        # Else just plot all data together
        else:
            ax.plot(self.timing_data.epochs, y_data, color='#0033A0')
        ax.set_xlabel('Epochs')
        if subtract_lin_params:
            ax.set_ylabel('Lin-Subtracted Mid-Times (Seconds)')
        else:
            ax.set_ylabel('Mid-Times (JD TDB)')
        ax.set_title(f'{model_data["model_type"].capitalize()} Model Mid-Times')
        ax.grid(linestyle='--', linewidth=0.25, zorder=-1)
        ax.ticklabel_format(style="plain", useOffset=False)
        if save_plot == True:
            fig.savefig(save_filepath, bbox_inches='tight', dpi=300)
        return ax

    def plot_timing_uncertainties(self, model_data, save_plot=False, save_filepath=None):
        """Plots a scatterplot of uncertainties on the fit model for each epoch.

        Parameters
        ----------
            model_data: dict
                The model data dictionary recieved from the `fit_model` method.
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        # get uncertainties
        fig, ax = plt.subplots(figsize=(6*(16/9), 6))
        uncertainties = self.get_model_uncertainties(model_data)
        x = self.timing_data.epochs
        # get T(E)-T0-PE (for transits), T(E)-T0-0.5P-PE (for occultations)
        y = self._subtract_linear_parameters(model_data['model_data'], model_data['conjunction_time'], model_data['period'], self.timing_data.epochs, self.timing_data.tra_or_occ)
        # plot the y line, then the line +- the uncertainties
        ax.plot(x, y, c='blue', label='$t(E) - T_{0} - PE$')
        ax.fill_between(x, y-uncertainties, y+uncertainties, alpha=0.2, label='$(t(E) - T_{0} - PE) \pm σ_{t^{pred}_{tra}}$')
        # ax.plot(x, y + uncertainties, c='red', label='$(t(E) - T_{0} - PE) + σ_{t^{pred}_{tra}}$')
        # ax.plot(x, y - uncertainties, c='red', label='$(t(E) - T_{0} - PE) - σ_{t^{pred}_{tra}}$')
        # Add labels and show legend
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mid-Time Uncertainties (JD TDB)')
        ax.set_title(f'Uncertainties of {model_data["model_type"].capitalize()} Model Mid-Times')
        ax.legend()
        ax.grid(linestyle='--', linewidth=0.25, zorder=-1)
        if save_plot is True:
            fig.savefig(save_filepath, bbox_inches='tight', dpi=300)
        return ax

    def plot_oc_plot(self, model_type, save_plot=False, save_filepath=None):
        """Plots a scatter plot of observed minus calculated mid-times. 
        
        Subtracts the linear portion from the observed mid-times. The linear portion is calculated using
        the best-fit parameters from the given model_type. 

        Parameters
        ----------
            model_type: str
                Either "quadratic" or "precession". One of the ephemerides being compared to the linear model.
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        fig, ax = plt.subplots(figsize=(6*(16/9), 6))
        DAYS_TO_SECONDS = 86400
        # y = T0 - PE - 0.5 dP/dE E^2
        linear_model = self.fit_model("linear")
        model = self.fit_model(model_type)
        # plot observed points w/ x=epoch, y=T(E)-T0-PE, yerr=sigmaT0
        y = (self._subtract_linear_parameters(self.timing_data.mid_times, linear_model['conjunction_time'], linear_model['period'], self.timing_data.epochs, self.timing_data.tra_or_occ)) * DAYS_TO_SECONDS
        self.oc_vals = y
        ax.errorbar(self.timing_data.epochs, y, yerr=self.timing_data.mid_time_uncertainties*DAYS_TO_SECONDS, 
                    marker='o', ls='', color='#0033A0',
                    label=r'$t(E) - T_0 - P E$')
        if model_type == "quadratic":
            # Plot additional quadratic curve
            # y = 0.5 dP/dE * (E - median E)^2
            quad_curve = (0.5*model['period_change_by_epoch'])*((self.timing_data.epochs - np.median(self.timing_data.epochs))**2) * DAYS_TO_SECONDS
            ax.plot(self.timing_data.epochs,
                    (quad_curve),
                    color='#D64309', ls="--", label=r'$\frac{1}{2}(\frac{dP}{dE})E^2$')
        if model_type == "precession":
            # Plot additional precession curve
            tra_mask = self.timing_data.tra_or_occ == "tra"
            occ_mask = self.timing_data.tra_or_occ == "occ"
            precession_curve_tra = (-1*((model["eccentricity"] * (model["period"] / (1 - ((1/(2*np.pi)) * model["pericenter_change_by_epoch"])))) / np.pi)*(np.cos(model["pericenter"] + (model["pericenter_change_by_epoch"] * (self.timing_data.epochs[tra_mask] - np.median(self.timing_data.epochs[tra_mask])))))) * DAYS_TO_SECONDS
            precession_curve_occ = (((model["eccentricity"] * (model["period"] / (1 - ((1/(2*np.pi)) * model["pericenter_change_by_epoch"])))) / np.pi)*(np.cos(model["pericenter"] + (model["pericenter_change_by_epoch"] * (self.timing_data.epochs[occ_mask] - np.median(self.timing_data.epochs[occ_mask])))))) * DAYS_TO_SECONDS
            ax.plot(self.timing_data.epochs[tra_mask],
                    (precession_curve_tra),
                    color='#D64309', ls="--", label=r'$-\frac{eP_a}{\pi}\cos\omega(E)$')
                # $\frac{1}{2}(\frac{dP}{dE})E^2$
            ax.plot(self.timing_data.epochs[occ_mask],
                    (precession_curve_occ),
                    color='#D64309', ls=":", label=r'$\frac{eP_a}{\pi}\cos\omega(E)$')
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('O-C (seconds)')
        ax.set_title(f'Observed Minus {model_type.capitalize()} Model Calculated Mid-Times')
        ax.grid(linestyle='--', linewidth=0.25, zorder=-1)
        if save_plot is True:
            fig.savefig(save_filepath, bbox_inches='tight', dpi=300)
        return ax

    def plot_running_delta_bic(self, model1, model2, save_plot=False, save_filepath=None):
        """Plots a scatterplot of epochs vs. :math:`\\Delta BIC` for each epoch.

        Starting at the third epoch, will plot the value of :math:`\\Delta BIC` for all previous epochs,
        showing how the value of :math:`\\Delta BIC` progresses over time with more observations.

        Parameters
        ----------
            model1: str
                Either "linear", "quadratic", or "precession". One of the ephemerides being compared.
            model2: str
                Either "linear", "quadratic", or "precession". One of the ephemerides being compared.
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        # Create empty array to store values
        delta_bics = np.zeros(len(self.timing_data.epochs))
        uncertainties = np.zeros(len(self.timing_data.epochs))
        # Create copy of each variable to be used
        all_epochs = self.timing_data.epochs.copy()
        all_mid_times = self.timing_data.mid_times.copy()
        all_uncertainties = self.timing_data.mid_time_uncertainties.copy()
        all_tra_or_occ = self.timing_data.tra_or_occ.copy()
        # For each epoch, calculate delta BIC using all data up to that epoch
        ks = (self._get_k_value(model1), self._get_k_value(model2))
        for i in range(max(ks), len(self.timing_data.epochs)):
            timing_data = TimingData("jd", all_epochs[:i+1], all_mid_times[:i+1], all_uncertainties[:i+1], all_tra_or_occ[:i+1], "tdb")
            # Create new model object with new timing data
            model = Ephemeris(timing_data)
            # Get delta bic of the two models
            delta_bic = model.calc_delta_bic(model1, model2)
            delta_bics[i] = delta_bic
            # Get uncertainties of the delta bic
            uncertainties[i] = np.sqrt((2*i-sum(ks)))
        # Plot the data
        fig, ax = plt.subplots(figsize=(6*(16/9), 6))
        ax.plot(self.timing_data.epochs, delta_bics, color='#0033A0', ls="-", linewidth=2, label=rf"$\Delta$BIC=BIC$_{{{model1}}}$ - BIC$_{{{model2}}}$={delta_bics[-1]:.2f}")
        ax.fill_between(self.timing_data.epochs, delta_bics-uncertainties, delta_bics+uncertainties, alpha=0.2)
        # ax.plot(self.timing_data.epochs, delta_bics, color='#0033A0', marker='.', markersize=6, mec="#D64309", ls="--", linewidth=2)
        ax.axhline(y=0, color='grey', linestyle='-', zorder=0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r"$\Delta$BIC")
        ax.set_title(rf"Value of $\Delta$BIC Comparing {model1.capitalize()} and {model2.capitalize()} Models"
                    "\n"
                    rf"as Observational Epochs Increase")
        ax.grid(linestyle='--', linewidth=0.25, zorder=-1)
        ax.legend()
        # Save if save_plot and save_filepath have been provided
        if save_plot and save_filepath:
            fig.savefig(save_filepath, bbox_inches='tight', dpi=300)
        # Return the axis (so it can be further edited if needed)
        return ax
    
    def plot_delta_bic_omit_one(self, model1="linear", model2="quadratic", outlier_percentage=None, save_plot=False, save_filepath=None):
        """
        Parameters
        ----------
            model1: str

            model2: str

            outlier_percentage: int

            save_plot: bool

            save_filepath: str

        """
        # Need to remove the data point at each index and calculate the BIC value
        # Plotting the BIC value when the point at each epoch is removed
        # If outlier percentage: calculate a percentage for each
        delta_bic = self.calc_delta_bic(model1, model2)
        delta_bics = np.zeros(len(self.timing_data.epochs))
        delta_bic_percentages = np.zeros(len(self.timing_data.epochs))
        for i in range(len(self.timing_data.epochs)):
            # Create new timing data with elements at the given index removed
            epochs = np.delete(self.timing_data.epochs, i)
            mid_times = np.delete(self.timing_data.mid_times, i)
            mid_time_uncertainties = np.delete(self.timing_data.mid_time_uncertainties, i)
            tra_or_occs = np.delete(self.timing_data.tra_or_occ, i)
            timing_data = TimingData("jd", epochs, mid_times, mid_time_uncertainties, tra_or_occs, "tdb")
            # Create new model object with new timing data
            model = Ephemeris(timing_data)
            # Get delta BIC
            d_bic = model.calc_delta_bic(model1, model2)
            delta_bics[i] = (d_bic)
            # Calculate the percentage difference 
            perc_diff = (abs(d_bic - delta_bic) / ((d_bic + delta_bic) / 2)) * 100
            delta_bic_percentages[i] = (perc_diff)
        fig, ax = plt.subplots(figsize=(6*(16/9), 6))
        ax.scatter(self.timing_data.epochs, delta_bics, color="#0033A0")
        ax.axhline(y=self.calc_delta_bic(model1, model2), color='#D64309', linestyle='--', zorder=0, label=rf"$\Delta$BIC = BIC$_{{{model1}}}$ - BIC$_{{{model2}}}$ = {self.calc_delta_bic(model1, model2):.2f}")
        # If we are given a percentage difference to mark, then plot that
        if outlier_percentage is not None:
            is_outlier = delta_bic_percentages >= outlier_percentage
            ax.scatter(self.timing_data.epochs[is_outlier], delta_bics[is_outlier], color="red", marker="s", zorder=10, label=rf"Shifts $\Delta$ BIC by ≥ {outlier_percentage}%")
        ax.set_xlabel("Epochs")
        ax.set_ylabel(r"$\Delta$BIC")
        ax.set_title(r"Final $\Delta$BIC if we Omit One Point")
        ax.grid(linestyle='--', linewidth=0.25, zorder=-1)
        ax.legend()
        # Save if save_plot and save_filepath have been provided
        if save_plot and save_filepath:
            fig.savefig(save_filepath, bbox_inches='tight', dpi=300)
        return ax

    def plot_running_analytical_delta_bic_quadratic(self, save_plot=False, save_filepath=None):
        """
            delta_bic = 0.25(dPdE)^2 * sum(()^2) - ln(N) + 1
        """
        # Grab a copy of all timing data
        all_epochs = self.timing_data.epochs[self.tra_mask].copy()
        all_mid_times = self.timing_data.mid_times[self.tra_mask].copy()
        all_uncertainties = self.timing_data.mid_time_uncertainties[self.tra_mask].copy()
        all_tra_or_occ = self.timing_data.tra_or_occ[self.tra_mask].copy()
        # Create some empty arrays to hold data
        numerical_delta_bics = np.zeros(len(self.timing_data.epochs[self.tra_mask]))
        numerical_delta_bic_errs = np.zeros(len(self.timing_data.epochs[self.tra_mask]))
        analytical_delta_bics = np.zeros(len(self.timing_data.epochs[self.tra_mask]))
        analytical_delta_bic_errs = np.zeros(len(self.timing_data.epochs[self.tra_mask]))
        # Grab the tidal decay rate for the analytical calculation
        dPdE = self.fit_model("quadratic")["period_change_by_epoch"]
        for i in range(3, len(self.timing_data.epochs[self.tra_mask])):
            timing_data = TimingData("jd", all_epochs[:i], all_mid_times[:i], all_uncertainties[:i], all_tra_or_occ[:i], "tdb")
            # Create new model object with new timing data
            model = Ephemeris(timing_data)
            # quad_model = model.fit_model("quadratic")
            # Calculate the numerical delta BIC
            numerical_delta_bic = model.calc_delta_bic("linear", "quadratic")
            numerical_delta_bics[i] = numerical_delta_bic
            # Calculate the analytical delta BIC
            analytical_delta_bic = model._calc_analytical_delta_bic_quad(dPdE)
            analytical_delta_bics[i] = analytical_delta_bic
            # Calculate the err
            numerical_delta_bic_errs[i] = np.sqrt((2*(i-5)))
            analytical_delta_bic_errs[i] = np.sqrt((2*(i-5)))
        # Plot the data
        fig, ax = plt.subplots(figsize=(6*(16/9), 6))
        ax.plot(self.timing_data.epochs[self.tra_mask], numerical_delta_bics, color='#0033A0', marker='.', markersize=7, mec="#0033A0", ls="-", linewidth=1.5, label=r"Evolution of Numerical $\Delta$BIC as Observational Epochs Increase")
        ax.plot(self.timing_data.epochs[self.tra_mask], analytical_delta_bics, color="#D64309", marker='.', markersize=7, mec="#D64309", ls="--", linewidth=1.5, label=r"Evolution of Analytical $\Delta$BIC as Observational Epochs Increase")
        ax.fill_between(self.timing_data.epochs[self.tra_mask], numerical_delta_bics-numerical_delta_bic_errs, numerical_delta_bics+numerical_delta_bic_errs, alpha=0.2)
        ax.fill_between(self.timing_data.epochs[self.tra_mask], analytical_delta_bics-analytical_delta_bic_errs, analytical_delta_bics+analytical_delta_bic_errs, alpha=0.2)
        ax.axhline(y=0, color='grey', linestyle='-', zorder=0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r"$\Delta$BIC")
        ax.set_title(rf"Evolution of $\Delta$BIC Comparing Linear and Quadratic Models")
        ax.grid(linestyle='--', linewidth=0.25, zorder=-1)
        ax.legend()
        # Save if save_plot and save_filepath have been provided
        if save_plot and save_filepath:
            fig.savefig(save_filepath, bbox_inches='tight', dpi=300)
        # Return the axis (so it can be further edited if needed)
        return ax

    def plot_running_analytical_delta_bic_precession(self, save_plot=False, save_filepath=None):
        """
        Plot the running analytical delta BIC

        NOTES: Using synthetic precession data generated from Brian's nb,
        Brian's parameters:
         'period': 1.091419633,
         'conjunction_time': 2456305.45488,
         'eccentricity': 0.0031,
         'pericenter': 2.62,
         'pericenter_change_by_epoch': 0.000984
        precession model parameters:
         'period': 1.0914194950612286, 
         'conjunction_time': 2456305.4547539037, 
         'eccentricity': 0.004492419251877544,
         'pericenter': 3.748694994358649,
         'pericenter_change_by_epoch': 0.0005248792790545403

        """
        # Create copy of each variable to be used
        all_epochs = self.timing_data.epochs[self.tra_mask].copy()
        all_mid_times = self.timing_data.mid_times[self.tra_mask].copy()
        all_uncertainties = self.timing_data.mid_time_uncertainties[self.tra_mask].copy()
        all_tra_or_occ = self.timing_data.tra_or_occ[self.tra_mask].copy()
        # Create some empty arrays to hold data
        numerical_delta_bics = np.zeros(len(self.timing_data.epochs[self.tra_mask]))
        analytical_delta_bics = np.zeros(len(self.timing_data.epochs[self.tra_mask]))
        # Grab precession model to calcualte the cos_ws
        precession_model = self.fit_model("precession")
        e = precession_model["eccentricity"]
        w0 = precession_model["pericenter"]
        dwdE = precession_model["pericenter_change_by_epoch"]
        Pa = self._calc_anomalistic_period(precession_model["period"], dwdE)
        # Iterate through all data points
        for i in range(5, len(self.timing_data.epochs[self.tra_mask])):
            timing_data = TimingData("jd", all_epochs[:i], all_mid_times[:i], all_uncertainties[:i], all_tra_or_occ[:i], "tdb")
            # Create new model object with new timing data
            model = Ephemeris(timing_data)
            # Calculate the numerical delta BIC
            numerical_delta_bic = model.calc_delta_bic("linear", "quadratic")
            numerical_delta_bics[i] = numerical_delta_bic
            # Calculate the analytical delta BIC
            analytical_delta_bic = model._calc_analytical_delta_bic_prec(Pa, e, w0, dwdE)
            analytical_delta_bics[i] = analytical_delta_bic
        # Plot the data
        fig, ax = plt.subplots(figsize=(6*(16/9), 6))
        ax.plot(self.timing_data.epochs[self.tra_mask], numerical_delta_bics, color='#0033A0', marker='.', markersize=7, mec="#0033A0", ls="-", linewidth=1.5, label=r"Evolution of Numerical $\Delta$BIC as Observational Epochs Increase")
        ax.plot(self.timing_data.epochs[self.tra_mask], analytical_delta_bics, color="#D64309", marker='.', markersize=7, mec="#D64309", ls="--", linewidth=1.5, label=r"Evolution of Analytical $\Delta$BIC as Observational Epochs Increase")
        ax.axhline(y=0, color='grey', linestyle='-', zorder=0)
        ax.set_xlabel("Epochs")
        ax.set_ylabel(r"$\Delta$BIC")
        ax.set_title(r"Evolution of $\Delta$BIC Comparing Linear and Precession Ephemerides")
        ax.grid(linestyle='--', linewidth=0.25, zorder=-1)
        ax.legend()
        # Save if save_plot and save_filepath have been provided
        if save_plot and save_filepath:
            fig.savefig(save_filepath, bbox_inches='tight', dpi=300)
        return ax

if __name__ == '__main__':
    # STEP 1: Upload datra from file
    bjd_filepath = "../../example_data/wasp12b_tra_occ.csv"
    bjd_no_occs_filepath = "../../example_data/WASP12b_transit_ephemeris.csv"
    isot_filepath = "../../example_data/wasp12b_isot_utc.csv"
    jd_utc_filepath = "../../example_data/wasp12b_jd_utc.csv"
    jd_utc_no_occs_filepath = "../../example_data/wasp12b_jd_utc_tra.csv"
    test_filepath = "../../example_data/test_data.csv"
    precession_test_filepath = "../../example_data/precession_test_data.csv"
    wasp12_tra_only = "../../example_data/WASP12b_transit_ephemeris.csv"
    data = np.genfromtxt(bjd_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # bjd_data_no_occs = np.genfromtxt(bjd_no_occs_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # isot_data = np.genfromtxt(isot_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # jd_utc_data = np.genfromtxt(jd_utc_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # jd_utc_no_occs_data = np.genfromtxt(jd_utc_no_occs_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # test_data = np.genfromtxt(test_filepath, delimiter=',', names=True, dtype=None, encoding=None)

    # STEP 2: Break data up into epochs, mid-times, and error
    # STEP 2.5 (Optional): Make sure the epochs are integers and not floats
    tra_or_occs = data["tra_or_occ"] # Base tra_or_occs
    # tra_or_occs = None # Base tra_or_occs
    epochs = data["epoch"].astype('int') # Epochs with tra_or_occs
    # epochs_no_occs = bjd_data_no_occs["epochs"]
    mid_times = data["mid_time"] # BJD mid times
    mid_time_errs = data["mid_time_err"] # BJD mid time errs
    # print(f"epochs: {list(epochs)}")
    # print(f"mid_times: {list(mid_times)}")
    # print(f"mid_time_errs: {list(mid_time_errs)}")
    # print(f"tra_or_occ: {list(tra_or_occs)}")

    # STEP 3: Create new transit times object with above data
    """NOTE: ISOT (NO UNCERTAINTIES)"""
    # times_obj1 = TimingData('isot', epochs, mid_times, tra_or_occ=tra_or_occs, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    """NOTE: JD UTC"""
    # times_obj1 = TimingData('jd', epochs, mid_times, mid_time_errs, tra_or_occ=tra_or_occs, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    """NOTE: JD UTC NO UNCERTAINTIES"""
    # times_obj1 = TimingData('jd', epochs, mid_times, tra_or_occ=tra_or_occs, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    """NOTE: JD UTC NO UNCERTAINTIES NO TRA_OR_OCC"""
    # times_obj1 = TimingData('jd', epochs_no_occs, mid_times, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    """NOTE: JD TDB (BJD) NO TRA_OR_OCC"""
    # times_obj1 = TimingData('jd', epochs_no_occs, mid_times, mid_time_errs, time_scale='tdb')
    """NOTE: JD TDB (BJD) ALL INFO"""
    times_obj1 = TimingData('jd', epochs, mid_times, mid_time_errs, time_scale='tdb', tra_or_occ=tra_or_occs)
    
    # STEP 4: Create new model object with transit times object
    ephemeris_obj1 = Ephemeris(times_obj1)
    
    # STEP 5: Get model data & BIC values
    # LINEAR MODEL
    linear_model = ephemeris_obj1.fit_model('linear')
    print(linear_model)
    linear_model_uncertainties = ephemeris_obj1.get_model_uncertainties(linear_model)
    print(linear_model_uncertainties)
    lin_bic = ephemeris_obj1.calc_bic(linear_model)
    print(lin_bic)
    # QUADRATIC MODEL
    quad_model = ephemeris_obj1.fit_model('quadratic')
    print(quad_model)
    quad_model_uncertainties = ephemeris_obj1.get_model_uncertainties(quad_model)
    print(quad_model_uncertainties)
    quad_bic = ephemeris_obj1.calc_bic(quad_model)
    print(quad_bic)
    # PRECESSION MODEL
    precession_model = ephemeris_obj1.fit_model("precession")
    print(precession_model)
    prec_bic = ephemeris_obj1.calc_bic(precession_model)
    print(prec_bic)
    # STEP 5.5a: Get the delta BIC value for the linear and quadratic ephemerides
    delta_bic_lq = ephemeris_obj1.calc_delta_bic("linear", "quadratic")
    print(delta_bic_lq)
    # STEP 5.5b: Get the delta BIC value for the linear and precession ephemerides
    delta_bic_lp = ephemeris_obj1.calc_delta_bic("linear", "precession")
    print(delta_bic_lp)
    # STEP 5.5c: Get the delta BIC value for the quadratic and precession ephemerides
    delta_bic_qp = ephemeris_obj1.calc_delta_bic("quadratic", "precession")
    print(delta_bic_qp)
    
    # STEP 6: Show a plot of the model data
    ephemeris_obj1.plot_model(linear_model, save_plot=False)
    plt.show()
    ephemeris_obj1.plot_model(quad_model, save_plot=False)
    plt.show()
    ephemeris_obj1.plot_model(precession_model, save_plot=False)
    plt.show()

    # STEP 7: Uncertainties plot
    ephemeris_obj1.plot_timing_uncertainties(linear_model, save_plot=False)
    plt.show()
    ephemeris_obj1.plot_timing_uncertainties(quad_model, save_plot=False)
    plt.show()
    
    # STEP 8: O-C Plot
    ephemeris_obj1.plot_oc_plot("quadratic", save_plot=False)
    plt.show()
    ephemeris_obj1.plot_oc_plot("precession", save_plot=False)
    plt.show()

    # STEP 9: Plot running delta BICs
    ephemeris_obj1.plot_running_delta_bic("linear", "quadratic")
    plt.show()
    ephemeris_obj1.plot_running_delta_bic("linear", "precession")
    plt.show()
    ephemeris_obj1.plot_running_delta_bic("quadratic", "precession")
    plt.show()

    ephemeris_obj1.plot_delta_bic_omit_one("linear", "quadratic")
    plt.show()
    ephemeris_obj1.plot_delta_bic_omit_one("linear", "precession")
    plt.show()
    ephemeris_obj1.plot_delta_bic_omit_one("quadratic", "precession")
    plt.show()

    ephemeris_obj1.plot_running_analytical_delta_bic_quadratic()
    plt.show()

    # STEP 10: Get observing schedule
    # ephemeris_obj1._get_eclipse_duration("TrES-3 b")
    # observer_obj = ephemeris_obj1.create_observer_obj(timezone="US/Mountain", longitude=-116.208710, latitude=43.602,
    #                                                   elevation=821, name="BoiseState")
    # target_obj = ephemeris_obj1.create_target_obj("TrES-3")
    # obs_sch = ephemeris_obj1.get_observing_schedule(quad_model, "US/Mountain", observer_obj, target_obj, 10, 2, "2024-12-04", exoplanet_name="TrES-3")

    # TODO: Do timezone checks and stuff in observing sched 

    # STEP 9: Running delta BIC plot
    # ephemeris_obj1.plot_running_delta_bic(model1="linear", model2="precession", save_plot=False)
    # plt.show()

    # nea_data = ephemeris_obj1._get_eclipse_system_params("WASP-12 b", ra=None, dec=None)
    # # nea_data = ephemeris_obj1._query_nasa_exoplanet_archive("WASP-12 b", select_query="pl_ratror,pl_orbsmax,pl_imppar,pl_orbincl")
    # print(nea_data)
    # print(np.arcsin(0.3642601363) * 0.3474 * 24)
    # ephemeris_obj1.plot_delta_bic_omit_one(outlier_percentage=5)
    # plt.show()lta_bic_omit_one(outlier_percentage=5)
    # plt.show()