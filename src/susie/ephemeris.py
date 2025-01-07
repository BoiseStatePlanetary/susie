from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
from astropy.units import Quantity
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget, Observer, EclipsingSystem, AtNightConstraint, AltitudeConstraint, is_event_observable
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
# from susie.timing_data import TimingData # Use this for package pushes
from .timing_data import TimingData # Use this for running tests
# from timing_data import TimingData # Use this for running this file

class BaseModelEphemeris(ABC):
    """Abstract class that defines the structure of different model ephemeris classes."""
    @abstractmethod
    def fit_model(self, x, y, yerr, tra_or_occ):
        """Fits a model ephemeris to timing data.

        Defines the structure for fitting a model (linear, quadratic or precession) to timing data. 
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
                Indicates if the data is from a transit or occultation.

        Returns
        ------- 
            A dictionary containing fitted model parameters. 
        """
        pass


class LinearModelEphemeris(BaseModelEphemeris):
    """Subclass of BaseModelEphemeris that implements a linear fit."""
    def lin_fit(self, E, T0, P, tra_or_occ_enum):
        """Calculates a linear function with given data.

        Uses the equation 
         - (period * epochs + initial mid time) for transit observations 
         - (period * epochs + (initial mid time + (½ * period ))) for occultation observations 
        
        as a linear function for an LMfit Model.
        
        Parameters
        ----------
            E: numpy.ndarray[float]
                The epochs.
            T0: float
                The initial mid-time, also known as conjunction time.
            P: float
                The exoplanet orbital period.
            tra_or_occ_enum: numpy.ndarray[int]
                Indicates if the data is from a transit or occultation. Enumerated to use 0 for transits and 1 for occultations.
        
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
    
    def fit_model(self, x, y, yerr, tra_or_occ):
        """Fits a linear model to ephemeris data.

        Compares the model ephemeris data to the linear fit created by data in TimingData object calculated 
        with lin_fit method. Then minimizes the difference between the two sets of data. LMfit Model then 
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
                Indicates if each point of data is taken from a transit or an occultation.

        Returns
        ------- 
        return_data: dict
            A dictionary of parameters from the fit model ephemeris.
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
        # TODO: Should we set this as the base estimate for T0 and P or should we try to find a good estimate to start with?
        params = model.make_params(T0=0.0, P=1.091423, tra_or_occ_enum=tra_or_occ_enum)
        result = model.fit(y, params, weights=1.0/yerr, E=x, tra_or_occ_enum=tra_or_occ_enum)
        return_data = {
            "period": result.params["P"].value,
            "period_err": result.params["P"].stderr,
            "conjunction_time": result.params["T0"].value,
            "conjunction_time_err": result.params["T0"].stderr
        }
        return(return_data)


class QuadraticModelEphemeris(BaseModelEphemeris):
    """Subclass of BaseModelEphemeris that implements a quadratic fit."""
    def quad_fit(self, E, T0, P, dPdE, tra_or_occ_enum):
        """Calculates a quadratic function with given data.

        Uses the equation 
         - ((0.5 * change in period over epoch * (epoch²)) + (period * epoch) + conjunction time) for transit observations
         - ((0.5 * change in period over epoch * (epoch²)) + (period * epoch) + conjunction time) for occultation observations as a quadratic function for the LMfit Model.
        
        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs.
            T0: float
                The initial mid-time, also known as conjunction time.
            P: float
                The exoplanet orbital period.
            dPdE: float
                Change in period with respect to epoch.
            tra_or_occ_enum: numpy.ndarray[int]
                Indicates if the data is from a transit or occultation. Enumerated to use 0 for transits and 1 for occultations.
        
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
    
    def fit_model(self, x, y, yerr, tra_or_occ):
        """Fits a quadratic model to ephemeris data.

        Compares the model ephemeris data to the quadratic fit calculated with quad_fit method. Then minimizes 
        the difference between the two sets of data. The LMfit Model then returns the parameters of the quadratic 
        function corresponding to period, conjunction time, period change by epoch, and their respective errors. 
        These parameters are returned in a dictionary to the user.

        Parameters
        ----------
            x: numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            y: numpy.ndarray[float]
                The mid-time data as recieved from the TimingData object.
            yerr: numpy.ndarray[float]
                The mid-time error data as recieved from the TimingData object.
            tra_or_occ: numpy.ndarray[str]
                Indicates if each point of data is taken from a transit or an occultation.

        Returns
        ------- 
        return_data: dict
            A dictionary of parameters from the fit model ephemeris. 
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
        # TODO: Should we set this as the base estimate for T0 and P or should we try to find a good estimate to start with?
        params = model.make_params(T0=0.0, P=1.091423, dPdE=0.0, tra_or_occ_enum=tra_or_occ_enum)
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
    

class PrecessionModelEphemeris(BaseModelEphemeris):
    """ Subclass of BaseModelEphemeris that implements a precession fit."""
    def _anomalistic_period(self, P, dwdE):
       """Calculates the anomalistic period given a period and a change in pericenter with respect to epoch.

       Uses the equation:
       P / (1 - (1/(2*pi)) * dwdE)

       Parameters
       ----------
       P: float
           The exoplanet sideral orbital period.
        dwdE: float
           Change in pericenter with respect to epoch.

        Returns
        -------
           A float of the calculated starting anomalistic period.
       """
       result = P/(1 - ((1/(2*np.pi))*dwdE))
       return result
    
    def _pericenter(self, E, w0, dwdE):
       """Calculates the pericenter given a list of epochs, an intial pericenter value, and a change in pericenter with respect to epoch.

       Uses the equation:
        w0 + dwdE * E

       Parameters
       ----------
        E: numpy.ndarray[int]
            The epochs.
        dwdE: float
            Change in pericenter with respect to epoch.
        w0: int
            The intial pericenter.

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
                The epochs.
            T0: float
                The initial mid-time, also known as conjunction time.
            P: float
                The exoplanet sideral orbital period.
            dwdE: float
                Change in pericenter with respect to epoch.
            w0: int
                The intial pericenter.
            e: float
                The eccentricity.
            tra_or_occ_enum: numpy.ndarray[int]
                Indicates if the data is from a transit or occultation. Enumerated to use 0 for transits and 1 for occultations.
        
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

    def fit_model(self, x, y, yerr, tra_or_occ):
        """Fits a precession model to ephemeris data.

        Compares the model ephemeris data to the precession fit calculated with precession_fit method. Then minimizes 
        the difference between the two sets of data. The LMfit Model then returns the parameters of the precession
        function corresponding to period, conjunction time, pericenter change by epoch, eccentricity, pericenter, and their respective errors. 
        These parameters are returned in a dictionary to the user.

        Parameters
        ----------
            x: numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            y: numpy.ndarray[float]
                The mid-time data as recieved from the TimingData object.
            yerr: numpy.ndarray[float]
                The mid-time error data as recieved from the TimingData object.
            tra_or_occ: numpy.ndarray[str]
                Indicates if each point of data is taken from a transit or an occultation.

        Returns
        ------- 
        return_data: dict
            A dictionary of parameters from the fit model ephemeris. 
            Example:
                {
                 'period': Estimated orbital period of the exoplanet (in units of days),
                 'period_err': Uncertainty associated with orbital period (in units of days),
                 'conjunction_time': Time of conjunction of exoplanet transit or occultation,
                 'conjunction_time_err': Uncertainty associated with conjunction_time,
                 'pericenter_change_by_epoch': The exoplanet pericenter change with respect to epoch,
                 'pericenter_change_by_epoch_err': The uncertainties associated with pericenter_change_by_epoch,
                 'eccentricity': The exoplanet pericenter,
                 'eccentricity_err': The uncertainties associated with eccentricity,
                 'pericenter': The exoplanet inital pericenter value,
                 'pericenter_err': The uncertainties associated with pericenter.
                }
        """
        # STARTING VAL OF dwdE CANNOT BE 0, WILL RESULT IN NAN VALUES FOR THE MODEL
        tra_or_occ_enum = [0 if i == 'tra' else 1 for i in tra_or_occ]
        model = Model(self.precession_fit, independent_vars=["E", "tra_or_occ_enum"])
        params = model.make_params(T0=0.0, P=1.091423, e=dict(value=0.00310, min=0, max=1), w0=2.62, dwdE=dict(value=0.000984), tra_or_occ_enum=tra_or_occ_enum)
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


class ModelEphemerisFactory:
    """Factory class for selecting which type of ephemeris class (linear, quadratic or precession) to use."""
    @staticmethod
    def create_model(model_type, x, y, yerr, tra_or_occ):
        """Instantiates the appropriate BaseModelEphemeris subclass and runs fit_model method.

        Based on the given user input of model type (linear, quadratic or precession) the factory will create the 
        corresponding subclass of BaseModelEphemeris and run the fit_model method to recieve the model 
        ephemeris return data dictionary.
        
        Parameters
        ----------
            model_type: str
                The name of the model ephemeris to create, either 'linear', 'quadratic' or 'precession'.
            x: numpy.ndarray[int]
                The epoch data as recieved from the TimingData object.
            y: numpy.ndarray[float]
                The mid-time data as recieved from the TimingData object.
            yerr: numpy.ndarray[float]
                The mid-time error data as recieved from the TimingData object.
            tra_or_occ: numpy.ndarray[str]
                Indicates if each point of data is taken from a transit or an occultation.

        Returns
        ------- 
            Model : dict
                A dictionary of parameters from the fit model ephemeris. If a linear model was chosen, these parameters are:
                    * 'period': Estimated orbital period of the exoplanet (in units of days),
                    * 'period_err': Uncertainty associated with orbital period (in units of days),
                    * 'conjunction_time': Time of conjunction of exoplanet transit or occultation,
                    * 'conjunction_time_err': Uncertainty associated with conjunction_time
                If a quadratic model was chosen, the same variables are returned, and an additional parameter is included in the dictionary:
                    * 'period_change_by_epoch': The exoplanet period change with respect to epoch (in units of days),
                    * 'period_change_by_epoch_err': The uncertainties associated with period_change_by_epoch (in units of days)
                If a precession model was chosen, the same varibales as linear are returned, and additional parameter is included in the dictionary:
                    * 'pericenter_change_by_epoch': The exoplanet pericenter change with respect to epoch,
                    * 'pericenter_change_by_epoch_err': The uncertainties associated with pericenter_change_by_epoch,
                    * 'eccentricity': The exoplanet pericenter,
                    * 'eccentricity': The uncertainties associated with eccentricity,
                    * 'pericenter': The exoplanet inital pericenter value,
                    * 'pericenter_err': The uncertainties associated with pericenter
        
        Raises
        ------
            ValueError:
                If model specified is not a valid subclass of BaseModelEphemeris, which is either 'linear', 'quadratic', or 'precession'.
        """
        models = {
            'linear': LinearModelEphemeris(),
            'quadratic': QuadraticModelEphemeris(),
            'precession': PrecessionModelEphemeris()
        }
        if model_type not in models:
            raise ValueError(f"Invalid model type: {model_type}")
        model = models[model_type]
        return model.fit_model(x, y, yerr, tra_or_occ)


class Ephemeris(object):
    """Represents the model ephemeris using transit or occultation mid-time data over epochs.

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
        """Initializing the model ephemeris object

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
                Indicates if each point of data is taken from a transit or an occultation.
        """
        x = self.timing_data.epochs
        y = self.timing_data.mid_times
        yerr = self.timing_data.mid_time_uncertainties
        tra_or_occ = self.timing_data.tra_or_occ
        return x, y, yerr, tra_or_occ
    
    def _get_model_parameters(self, model_type, **kwargs):
        """Creates the model ephemeris object and returns model parameters.
        
        This method fetches data from the TimingData object to be used in the model ephemeris. 
        It creates the appropriate subclass of BaseModelEphemeris using the ModelEphemeris factory, then runs 
        the fit_model method to return the model parameters dictionary to the user.

        Parameters
        ----------
            model_type: str
                Either 'linear', 'quadratic', or 'precession'. The ephemeris subclass specified to create and run.

        Returns
        -------
            model_ephemeris_data: dict
                A dictionary of parameters from the fit model ephemeris. 
                If a linear model was chosen, these parameters are:
                {
                    'period': Estimated orbital period of the exoplanet (in units of days),
                    'period_err': Uncertainty associated with orbital period (in units of days),
                    'conjunction_time': Time of conjunction of exoplanet transit or occultation,
                    'conjunction_time_err': Uncertainty associated with conjunction_time
                }
                If a quadratic model was chosen, the same variables are returned, and an additional parameter is included in the dictionary:
                {
                    'period_change_by_epoch': The exoplanet period change with respect to epoch (in units of days),
                    'period_change_by_epoch_err': The uncertainties associated with period_change_by_epoch (in units of days)
                }
                If a precession model was chosen, the same variables as the linear model are returned, and additional parameters are included in the dictionary:
                {
                    'period': Estimated orbital period of the exoplanet (in units of days),
                    'period_err': Uncertainty associated with orbital period (in units of days),
                    'conjunction_time': Time of conjunction of exoplanet transit or occultation,
                    'conjunction_time_err': Uncertainty associated with conjunction_time,
                    'pericenter_change_by_epoch': The exoplanet pericenter change with respect to epoch,
                    'pericenter_change_by_epoch_err': The uncertainties associated with pericenter_change_by_epoch,
                    'eccentricity': The exoplanet pericenter,
                    'eccentricity_err': The uncertainties associated with eccentricity,
                    'pericenter': The exoplanet inital pericenter value,
                    'pericenter_err': The uncertainties associated with pericenter.
                }

        Raises
        ------
            ValueError:
                If model specified is not a valid subclass of BaseModelEphemeris, which is either 'linear', 'quadratic', or 'precession'.
        """
        # Step 1: Get data from transit times obj
        x, y, yerr, tra_or_occ = self._get_timing_data()
        # Step 2: Create the model with the given variables & user inputs. 
        # This will return a dictionary with the model parameters as key value pairs.
        model_ephemeris_data = ModelEphemerisFactory.create_model(model_type, x, y, yerr, tra_or_occ, **kwargs)
        # Step 3: Return the data dictionary with the model parameters
        return model_ephemeris_data
    
    def _get_k_value(self, model_type):
        """Returns the number of parameters value to be used in the BIC calculation.
        
        Parameters
        ----------
            model_type: str
                Either 'linear' or 'quadratic', used to specify how many fit parameters are present in the model.

        Returns
        -------
            An int representing the number of fit parameters for the model. This will be 2 for a linear ephemeris 
            and 3 for a quadratic ephemeris.

        Raises
        ------
            ValueError
                If the model_type is an unsupported model type. Currently supported model types are 'linear' and 
                'quadratic'.
        """
        if model_type == 'linear':
            return 2
        elif model_type == 'quadratic':
            return 3
        elif model_type == 'precession':
            return 5
        else:
            raise ValueError('Only linear, quadratic, and precession models are supported at this time.')
    
    def _calc_anomalistic_period(self, P, dwdE):
       """ Calculates the anomalistic period given a period and a change in pericenter with respect to epoch.

       Uses the equation
       :math:`\\frac{P}{(1 - \\frac{1}{2*\\pi})*frac{dw}{dE}}`
       
       Parameters
       ----------
       P: float
           The exoplanet sideral orbital period.
        dwdE: float
           Change in pericenter with respect to epoch.

        Returns
        -------
           A float of the calculated anomalistic period.
       """
       result = P/(1 - (1/(2*np.pi))*dwdE)
       return result
    
    def _calc_pericenter(self, w0, dwdE, E):
       """Calculates the pericenter given a list of epochs, a pericenter value, and a change in pericenter with respect to epoch.

       Uses the equation
        :math:`w0 + \\frac{dw}{dE} * E`


       Parameters
       ----------
        E: numpy.ndarray[int]
            The epochs.
        dwdE: float
            Change in pericenter with respect to epoch.
        w0: int
            The pericenter.

        Returns
        -------
           A numpy.ndarray[float] of the calculated pericenter as a function of epochs.
       """
       result = w0 + dwdE*E
       return result
    
    def _calc_linear_model_uncertainties(self, T0_err, P_err):
        """Calculates the uncertainties of a given linear model when compared to actual data in TimingData.
        
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
        between the model data and actual data over epochs.
        
        Parameters
        ----------
        T0_err: numpy.ndarray[float]
            The calculated conjunction time error from a linear model ephemeris.
        P_err: numpy.ndarray[float]
            The calculated period error from a linear model ephemeris.
        
        Returns
        -------
            A list of uncertainties associated with the model ephemeris data passed in, calculated with the 
            equation above and the TimingData epochs.
        """
        result = np.zeros(len(self.timing_data.epochs))
        result[self.tra_mask] = np.sqrt((T0_err**2) + ((self.timing_data.epochs[self.tra_mask]**2)*(P_err**2)))
        result[self.occ_mask] = np.sqrt((T0_err**2) + (((self.timing_data.epochs[self.occ_mask]+0.5)**2)*(P_err**2)))
        return result
    
    def _calc_quadratic_model_uncertainties(self, T0_err, P_err, dPdE_err):
        """Calculates the uncertainties of a given quadratic model when compared to actual data in TimingData.
        
        Uses the equation 
         - .. math::
            \\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * E^2) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))} 
            for transit observations
         - .. math::
            \\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * (\\frac{1}{2} + E^2)) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))} 
            for occultation observations

         - σ(t pred, tra) = √(σ(T0)² + (σ(P)² * E²) + (¼ * σ(dP/dE)² * E⁴)) for transit observation
         - σ(t pred, occ) = √(σ(T0)² + (σ(P)² * (½ + E)²) + (¼ * σ(dP/dE)² * E⁴)) for occultation observations
        where σ(T0)=conjunction time error, E=epoch, σ(P)=period error, and σ(dP/dE)=period change with respect 
        to epoch error, to calculate the uncertainties between the model data and actual data over epochs.
        
        Parameters
        ----------
        T0_err: numpy.ndarray[float]
            The calculated conjunction time error from a quadratic model ephemeris.
        P_err: numpy.ndarray[float]
            The calculated period error from a quadratic model ephemeris.
        dPdE_err: numpy.ndarray[float]
            The calculated change in period with respect to epoch error for a quadratic model ephemeris.
        
        Returns
        -------
            A list of uncertainties associated with the model ephemeris passed in, calculated with the 
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
    
    def _calc_linear_ephemeris(self, E, T0, P):
        """Calculates mid-times using parameters from a linear model ephemeris.
        
        Uses the equation:
         - (T0 + PE) for transit observations
         - ((T0 + ½P) + PE) for occultation observations
        to calculate the mid-time times over each epoch where T0 is conjunction time, P is period, 
        and E is epoch.

        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs pulled from the TimingData object.
            T0: float
                The conjunction time of the exoplanet as calculated by the linear ephemeris model.
            P: float
                The orbital period of the exoplanet as calculated by the linear ephemeris model.

        Returns
        -------
            A numpy array of mid-time times calculated over each epoch using the equation above.
        """
        result = np.zeros(len(self.timing_data.epochs))
        result[self.tra_mask] = T0 + (P*E[self.tra_mask])
        result[self.occ_mask] = (T0 + 0.5*P) + (P*E[self.occ_mask])
        return result
    
    def _calc_quadratic_ephemeris(self, E, T0, P, dPdE):
        """Calculates mid-times using parameters from a quadratic model ephemeris.

        Uses the equation:
         - (T0 + PE + (½ * dPdE * E²)) for transit observations
         - ((T0 + ½P) + PE + (½ * dPdE * E²)) for occultation observations
        to calculate the mid-times over each epoch where T0 is conjunction time, P is period, E is epoch, 
        and dPdE is period change with respect to epoch.

        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs pulled from the TimingData object.
            T0: float
                The conjunction time of the exoplanet as calculated by the linear ephemeris model.
            P: float
                The orbital period of the exoplanet as calculated by the linear ephemeris model.
            dPdE: float
                The period change with respect to epoch as calculated by the linear ephemeris model.

        Returns
        -------
            A numpy array of mid-times calculated over each epoch using the equation above.
        """
        result = np.zeros(len(self.timing_data.epochs))
        result[self.tra_mask] = T0 + P*E[self.tra_mask] + 0.5*dPdE*E[self.tra_mask]*E[self.tra_mask]
        result[self.occ_mask] = (T0 + 0.5*P) + P*E[self.occ_mask] + 0.5*dPdE*E[self.occ_mask]*E[self.occ_mask]
        return result
    
    def _calc_precession_ephemeris(self, E, T0, P, e, w0, dwdE):
        """Calculates mid-times using parameters from a precession model ephemeris.

        Uses the equation:
         -  T0 + E*P - (e*anomalistic period)/pi * cos(pericenter) for transit observations
         -  T0 + (anomalistic period / 2) + (e*anomalistic period)/pi * cos(pericenter) for occultation observations
        to calculate the mid-times over each epoch where T0 is conjunction time, P is sideral period, E is epoch, 
        dwdE is pericenter change with respect to epoch, w0 is intial pericenter, e is eccentricity.

        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs.
            T0: float
                The initial mid-time, also known as conjunction time.
            P: float
                The exoplanet sideral orbital period.
            e: float
                The eccentricity.
            w0: int
                The argument of periastron (initial pericenter).
            dwdE: float
                The precession rate, or the change in pericenter with respect to epoch.
            tra_or_occ: numpy.ndarray[str]
                Indicates if the data is from a transit or occultation.

        Returns
        -------
            A numpy array of mid-times calculated over each epoch using the equation above.
        """ 
        result = np.zeros(len(self.timing_data.epochs))
        result[self.tra_mask] = T0 + E[self.tra_mask]*P - ((e*self._calc_anomalistic_period(P, dwdE))/np.pi)*np.cos(self._calc_pericenter(w0, dwdE, E[self.tra_mask]))
        result[self.occ_mask] = T0 + (self._calc_anomalistic_period(P, dwdE)/2) + (E[self.occ_mask]*P) + ((e*self._calc_anomalistic_period(P, dwdE))/np.pi)*np.cos(self._calc_pericenter(w0, dwdE, E[self.occ_mask]))
        return result
   
    def _calc_chi_squared(self, model_mid_times):
        """Calculates the residual chi squared values for the model ephemeris.
        
        Parameters
        ----------
            model_mid_times : numpy.ndarray[float]
                Mid-times calculated from a model ephemeris. This data can be accessed through the 'model_data'
                key from a returned model ephemeris data dictionary. 
        
        Returns
        -------
            Chi-squared value : float
                The chi-squared value calculated with the equation:
                Σ(((observed mid-times - model calculated mid-times) / observed mid-time uncertainties)²)
        """
        # STEP 1: Get observed mid-times
        observed_data = self.timing_data.mid_times
        uncertainties = self.timing_data.mid_time_uncertainties
        # STEP 2: calculate X2 with observed data and model data
        return np.sum(((observed_data - model_mid_times)/uncertainties)**2)
    
    def _subtract_linear_parameters(self, model_mid_times, T0, P, E, tra_or_occ):
        """Subtracts the linear terms to show smaller changes in other model parameters for plotting functions.

        Uses the equations:
         - (model midtime - T0 - PE) for transit observations
         - (model midtime - T0 - (½P) - PE) for occultation observations
        
        Parameters
        ----------
            model_mid_times : numpy.ndarray[float]
                Mid-times calculated from a model ephemeris. This data can be accessed through the 'model_data'
                key from a returned model ephemeris data dictionary. 
            T0: float
                The conjunction time of the exoplanet as calculated by the linear ephemeris model.
            P: float
                The orbital period of the exoplanet as calculated by the linear ephemeris model.
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
    
    def get_model_ephemeris(self, model_type):
        """Fits the timing data to a specified model using an LMfit Model fit.

        Parameters
        ----------
            model_type: str
                Either 'linear' or 'quadratic'. Represents the type of ephemeris to fit the data to.

        Returns
        ------- 
            model_ephemeris_data: dict
                A dictionary of parameters from the fit model ephemeris.
                    Example:

                    .. code-block:: python
            
                        {
                            'model_type': 'Either linear or quadratic',
                            'model_data': 'A list of calculated mid-times using the estimated parameters over each epoch',
                            'period': 'Estimated orbital period of the exoplanet (in units of days)',
                            'period_err': 'Uncertainty associated with orbital period (in units of days)',
                            'conjunction_time': 'Time of conjunction of exoplanet transit or occultation',
                            'conjunction_time_err': 'Uncertainty associated with conjunction_time',
                        }
                    
                    If a quadratic model was chosen, the same variables are returned, and an additional parameter is included in the dictionary:
                    
                    .. code-block:: python
                        
                        {
                            'period_change_by_epoch': 'The exoplanet period change with respect to epoch (in units of days)',
                            'period_change_by_epoch_err': 'The uncertainties associated with period_change_by_epoch (in units of days)'
                        }
        """
        model_ephemeris_data = self._get_model_parameters(model_type)
        model_ephemeris_data["model_type"] = model_type
        # Once we get parameters back, we call _calc_blank_ephemeris 
        if model_type == "linear":
            # Return dict with parameters and model data
            model_ephemeris_data["model_data"] = self._calc_linear_ephemeris(self.timing_data.epochs, model_ephemeris_data["conjunction_time"], model_ephemeris_data["period"])
        elif model_type == "quadratic":
            model_ephemeris_data["model_data"] = self._calc_quadratic_ephemeris(self.timing_data.epochs, model_ephemeris_data["conjunction_time"], model_ephemeris_data["period"], model_ephemeris_data["period_change_by_epoch"])
        elif model_type == "precession":
            model_ephemeris_data["model_data"] = self._calc_precession_ephemeris(self.timing_data.epochs, model_ephemeris_data["conjunction_time"], model_ephemeris_data["period"], model_ephemeris_data["eccentricity"], model_ephemeris_data["pericenter"], model_ephemeris_data["pericenter_change_by_epoch"])
        return model_ephemeris_data
    
    def get_ephemeris_uncertainties(self, model_params):
        """Calculates the mid-time uncertainties of specific model data when compared to the actual data. 

        Calculate the uncertainties between the model data and actual data over epochs using the equations...
        
        For linear models:
        
         - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * E^2)}` for transit observations

         - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * (\\frac{1}{2} + E)^2)}` for occultation observations
            
        And for quadratic models:

         - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * E^2) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))}` for transit observations

         - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * (\\frac{1}{2} + E^2)) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))}` for occultation observations
        
        where :math:`\\sigma(T_0) =` conjunction time error, :math:`E=` epoch, :math:`\\sigma(P)=` period error, and :math:`\\sigma(\\frac{dP}{dE})=` period change with respect to epoch error.
        
        Parameters
        ----------
        model_params: dict
            The model ephemeris data dictionary recieved from the `get_model_ephemeris` method.
        
        Returns
        -------
            A list of mid-time uncertainties associated with the model ephemeris passed in, calculated with the 
            equation above and the TimingData epochs.
        
        Raises
        ------
            KeyError
                If the model type in not in the model parameter dictionary.
            KeyError
                If the model parameter error values are not in the model parameter dictionary.
        """
        linear_params = ['conjunction_time', 'conjunction_time_err', 'period', 'period_err']
        quad_params = ['conjunction_time', 'conjunction_time_err', 'period', 'period_err', 'period_change_by_epoch', 'period_change_by_epoch_err']
        prec_params = ['conjunction_time', 'conjunction_time_err', 'period', 'period_err', 'pericenter_change_by_epoch', 'pericenter_change_by_epoch_err', 'eccentricity', 'eccentricity_err', 'pericenter', 'pericenter_err']
        if 'model_type' not in model_params:
            raise KeyError("Cannot find model type in model data. Please run the get_model_ephemeris method to return ephemeris fit parameters.")
        if model_params['model_type'] == 'linear':
            if any(linear_params) not in model_params:
                raise KeyError("Cannot find conjunction time and period and/or their respective errors in model data. Please run the get_model_ephemeris method with 'linear' model_type to return ephemeris fit parameters.")
            return self._calc_linear_model_uncertainties(model_params['conjunction_time_err'], model_params['period_err'])
        elif model_params['model_type'] == 'quadratic':
            if any(quad_params) not in model_params:
                raise KeyError("Cannot find conjunction time, period, and/or period change by epoch and/or their respective errors in model data. Please run the get_model_ephemeris method with 'quadratic' model_type to return ephemeris fit parameters.")
            return self._calc_quadratic_model_uncertainties(model_params['conjunction_time_err'], model_params['period_err'], model_params['period_change_by_epoch_err'])
        elif model_params['model_type'] == 'precession':
            if any(prec_params) not in model_params:
                raise KeyError("Cannot find conjunction time, period, eccentricity, pericenter, and/or pericenter change by epoch and/or their respective errors in model data. Please run the get_model_ephemeris method with 'precession' model_type to return ephemeris fit parameters.")
            return self._calc_precession_model_uncertainties(model_params)

    def calc_bic(self, model_data_dict):
        """Calculates the BIC value for a given model ephemeris. 
        
        The BIC value is a modified :math:`\\chi^2` value that penalizes for additional parameters. 
        Uses the equation :math:`BIC = \\chi^2 + (k * log(N))` where :math:`\\chi^2=\\sum{\\frac{(\\text{
        observed midtimes - model midtimes})}{\\text{(observed midtime uncertainties})^2}},`
        k=number of fit parameters (2 for linear models, 3 for quadratic models), and N=total number of data points.
        
        Parameters
        ----------
            model_data_dict: dict
                The model ephemeris data dictionary recieved from the `get_model_ephemeris` method.
        
        Returns
        ------- 
            A float value representing the BIC value for this model ephemeris.
        """
        # Step 1: Get value of k based on model_type (linear=2, quad=3, custom=?)
        num_params = self._get_k_value(model_data_dict['model_type'])
        # Step 2: Calculate chi-squared
        chi_squared = self._calc_chi_squared(model_data_dict['model_data'])
        # Step 3: Calculate BIC
        return chi_squared + (num_params*np.log(len(model_data_dict['model_data'])))

    def calc_delta_bic(self, model1, model2):
        """Calculates the :math:`\\Delta BIC` value between linear and quadratic model ephemerides using the given timing data. 
        
        The BIC value is a modified :math:`\\chi^2` value that penalizes for additional parameters. The
        :math:`\\Delta BIC` value is the difference between the linear BIC value and the quadratic BIC value.
        Models that have smaller values of BIC are favored. Therefore, if the :math:`\\Delta BIC` value for your
        data is a large positive number (large linear BIC - small quadratic BIC), the quadratic model is favored and
        your data indicates possible orbital decay in your extrasolar system. If the :math:`\\Delta BIC` value for
        your data is a small number or negative (small linear BIC - large quadratic BIC), then the linear model is
        favored and your data may not indicate orbital decay. 

        This function will run all model ephemeris instantiation and BIC calculations for you using the TimingData
        information you entered.

        Parameters
        ----------
            model1: str
                This is the name of the model you would like to compare to the linear model.
                Can be either "quadratic" or "precession".

        Returns
        ------- 
            delta_bic : float
                Represents the :math:`\\Delta BIC` value for this timing data. 
        """
        # For quad: lin - quad (model1="linear", model2="quadratic")
        # For prec: lin - prec (model1="linear", model2="precession")
        valid_models = ["linear", "quadratic", "precession"]
        if model1 not in valid_models or model2 not in valid_models:
            raise ValueError("Only linear, quadratic, and precession models are accepted at this time.")
        model1_data = self.get_model_ephemeris(model1)
        model2_data = self.get_model_ephemeris(model2)
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
                return val_to_store * u.day
            
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
    
    def get_observing_schedule(self, model_data_dict, timezone, observer, target, n_transits, n_occultations, obs_start_time, exoplanet_name=None, eclipse_duration=None):
        """Returns a list of observable future transits for the target object

        Parameters
        ----------
            model_data_dict: dict
                The model ephemeris data dictionary recieved from the `get_model_ephemeris` method.
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

            n_occultations: int

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
        """
        # Validate some things before continuing
        self._validate_observing_schedule_params(observer, target, obs_start_time)
        # Grab the most recent mid transit time
        primary_eclipse_time = Time(self.timing_data.mid_times[-1], format='jd')
        # Pull orbital period from the model
        orbital_period = model_data_dict['period'] * u.day
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
        # ing_egr_occultations = eclipsing_system.next_secondary_ingress_egress_time(obs_time, n_eclipses=n_occultations)
        # We need to check if the events are observable
        constraints = [AtNightConstraint.twilight_civil(), AltitudeConstraint(min=30*u.deg)]
        transits_bool = is_event_observable(constraints, observer, target, times=ing_egr_transits)
        # occultations_bool = is_event_observable(constraints, observer, target, times=ing_egr_occultations)
        print(transits_bool)
        # observable_transits = ing_egr_transits[transits_bool]
        # observable_occultations = ing_egr_occultations[occultations_bool]
        # print(observable_transits)
        # print(observable_occultations)
        pass    
    
    def plot_model_ephemeris(self, model_data_dict, subtract_lin_params=False, show_occultations=False, save_plot=False, save_filepath=None):
        """Plots a scatterplot of epochs vs. model calculated mid-times.

        Parameters
        ----------
            model_data_dict: dict
                The model ephemeris data dictionary recieved from the `get_model_ephemeris` method.
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        fig, ax = plt.subplots()
        y_data = model_data_dict['model_data']
        if subtract_lin_params is True:
            y_data = self._subtract_linear_parameters(model_data_dict['model_data'], model_data_dict['conjunction_time'], model_data_dict['period'], self.timing_data.epochs, self.timing_data.tra_or_occ)
        if show_occultations is True:
            occ_mask = self.timing_data.tra_or_occ == "occ"
            occ_data = model_data_dict["model_data"][occ_mask]
            if subtract_lin_params is True:
                occ_data = self._subtract_linear_parameters(occ_data, model_data_dict['conjunction_time'], model_data_dict['period'], self.timing_data.epochs[occ_mask], self.timing_data.tra_or_occ[occ_mask])
            ax.scatter(x=self.timing_data.epochs[occ_mask], y=occ_data, color="#D64309", zorder=20)
        ax.scatter(x=self.timing_data.epochs, y=y_data, color='#0033A0', zorder=10)
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mid-Times (JD TDB)')
        ax.set_title(f'{model_data_dict["model_type"].capitalize()} Model Ephemeris Mid-Times')
        ax.grid(linestyle='--', linewidth=0.25, zorder=-1)
        ax.ticklabel_format(style="plain", useOffset=False)
        if save_plot == True:
            fig.savefig(save_filepath, bbox_inches='tight', dpi=300)
        return ax

    def plot_timing_uncertainties(self, model_data_dict, save_plot=False, save_filepath=None):
        """Plots a scatterplot of epochs vs. model calculated mid-time uncertainties.

        Parameters
        ----------
            model_data_dict: dict
                The model ephemeris data dictionary recieved from the `get_model_ephemeris` method.
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        # get uncertainties
        fig, ax = plt.subplots()
        model_uncertainties = self.get_ephemeris_uncertainties(model_data_dict)
        x = self.timing_data.epochs
        # get T(E)-T0-PE (for transits), T(E)-T0-0.5P-PE (for occultations)
        y = self._subtract_linear_parameters(model_data_dict['model_data'], model_data_dict['conjunction_time'], model_data_dict['period'], self.timing_data.epochs, self.timing_data.tra_or_occ)
        # plot the y line, then the line +- the uncertainties
        ax.plot(x, y, c='blue', label='$t(E) - T_{0} - PE$')
        ax.plot(x, y + model_uncertainties, c='red', label='$(t(E) - T_{0} - PE) + σ_{t^{pred}_{tra}}$')
        ax.plot(x, y - model_uncertainties, c='red', label='$(t(E) - T_{0} - PE) - σ_{t^{pred}_{tra}}$')
        # Add labels and show legend
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Mid-Time Uncertainties (JD TDB)')
        ax.set_title(f'Uncertainties of {model_data_dict["model_type"].capitalize()} Model Ephemeris Mid-Times')
        ax.legend()
        ax.grid(linestyle='--', linewidth=0.25, zorder=-1)
        if save_plot is True:
            fig.savefig(save_filepath, bbox_inches='tight', dpi=300)
        return ax

    def plot_oc_plot(self, model, save_plot=False, save_filepath=None):
        """Plots a scatter plot of observed minus calculated values of mid-times for linear and quadratic model ephemerides over epochs.

        Parameters
        ----------
            model: str
                Either "quadratic" or "precession". One of the models being compared to the linear model.
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        fig, ax = plt.subplots()
        DAYS_TO_SECONDS = 86400
        # y = T0 - PE - 0.5 dP/dE E^2
        # lin_model_data = self.get_model_ephemeris("linear")
        model_data = self.get_model_ephemeris(model)
        # plot observed points w/ x=epoch, y=T(E)-T0-PE, yerr=sigmaT0
        y = (self._subtract_linear_parameters(self.timing_data.mid_times, model_data['conjunction_time'], model_data['period'], self.timing_data.epochs, self.timing_data.tra_or_occ)) * DAYS_TO_SECONDS
        self.oc_vals = y
        ax.errorbar(self.timing_data.epochs, y, yerr=self.timing_data.mid_time_uncertainties*DAYS_TO_SECONDS, 
                    marker='o', ls='', color='#0033A0',
                    label=r'$t(E) - T_0 - P E$')
        if model == "quadratic":
            # Plot additional quadratic curve
            # y = 0.5 dP/dE * (E - median E)^2
            quad_model_curve = (0.5*model_data['period_change_by_epoch'])*((self.timing_data.epochs - np.median(self.timing_data.epochs))**2) * DAYS_TO_SECONDS
            ax.plot(self.timing_data.epochs,
                    (quad_model_curve),
                    color='#D64309', ls="--", label=r'$\frac{1}{2}(\frac{dP}{dE})E^2$')
        if model == "precession":
            # Plot additional precession curve
            # y = -
            tra_mask = self.timing_data.tra_or_occ == "tra"
            occ_mask = self.timing_data.tra_or_occ == "occ"
            precession_model_curve_tra = (-1*((model_data["eccentricity"] * (model_data["period"] / (1 - ((1/(2*np.pi)) * model_data["pericenter_change_by_epoch"])))) / np.pi)*(np.sin(model_data["pericenter"] + (model_data["pericenter_change_by_epoch"] * (self.timing_data.epochs[tra_mask] - np.median(self.timing_data.epochs[tra_mask])))))) * DAYS_TO_SECONDS
            precession_model_curve_occ = (((model_data["eccentricity"] * (model_data["period"] / (1 - ((1/(2*np.pi)) * model_data["pericenter_change_by_epoch"])))) / np.pi)*(np.sin(model_data["pericenter"] + (model_data["pericenter_change_by_epoch"] * (self.timing_data.epochs[occ_mask] - np.median(self.timing_data.epochs[occ_mask])))))) * DAYS_TO_SECONDS
            ax.plot(self.timing_data.epochs[tra_mask],
                    (precession_model_curve_tra),
                    color='#D64309', ls="--", label=r'$-\frac{eP_a}{\pi}\cos\omega(E)$')
                # $\frac{1}{2}(\frac{dP}{dE})E^2$
            ax.plot(self.timing_data.epochs[occ_mask],
                    (precession_model_curve_occ),
                    color='#D64309', ls=":", label=r'$\frac{eP_a}{\pi}\cos\omega(E)$')
        ax.legend()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('O-C (seconds)')
        ax.set_title('Observed Minus Calculated Mid-Times')
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
                Either "linear", "quadratic", or "precession". One of the models being compared.
            model2: str
                Either "linear", "quadratic", or "precession". One of the models being compared.
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        # Create empty array to store values
        delta_bics = []
        # Create copy of each variable to be used
        all_epochs = self.timing_data.epochs.copy()
        all_mid_times = self.timing_data.mid_times.copy()
        all_uncertainties = self.timing_data.mid_time_uncertainties.copy()
        all_tra_or_occ = self.timing_data.tra_or_occ.copy()
        # For each epoch, calculate delta BIC using all data up to that epoch
        for i in range(0, len(all_epochs)):
            if i < max(self._get_k_value(model1), self._get_k_value(model2))-1:
                # Append 0s up until delta BIC can be calculated
                delta_bics.append(int(0))
            else:
                self.timing_data.epochs = all_epochs[:i+1]
                self.timing_data.mid_times = all_mid_times[:i+1]
                self.timing_data.mid_time_uncertainties = all_uncertainties[:i+1]
                self.timing_data.tra_or_occ = all_tra_or_occ[:i+1]
                delta_bic = self.calc_delta_bic(model1, model2)
                delta_bics.append(delta_bic)
        # Plot the data
        fig, ax = plt.subplots(figsize=(15, 7))
        ax.plot(self.timing_data.epochs, delta_bics, color='#0033A0', marker='.', markersize=6, mec="#D64309", ls="--", linewidth=2)
        ax.axhline(y=0, color='grey', linestyle='-', zorder=0)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(r"$\Delta$BIC")
        ax.set_title(rf"Value of $\Delta$BIC Comparing {model1.capitalize()} and {model2.capitalize()} Models"
                    "\n"
                    rf"as Observational Epochs Increase")
        ax.grid(linestyle='--', linewidth=0.25, zorder=-1)
        # Save if save_plot and save_filepath have been provided
        if save_plot and save_filepath:
            fig.savefig(save_filepath, bbox_inches='tight', dpi=300)
        # Return the axis (so it can be further edited if needed)
        return ax

if __name__ == '__main__':
    # STEP 1: Upload datra from file
    bjd_filepath = "../../example_data/wasp12b_tra_occ.csv"
    bjd_no_occs_filepath = "../../example_data/WASP12b_transit_ephemeris.csv"
    isot_filepath = "../../example_data/wasp12b_isot_utc.csv"
    jd_utc_filepath = "../../example_data/wasp12b_jd_utc.csv"
    jd_utc_no_occs_filepath = "../../example_data/wasp12b_jd_utc_tra.csv"
    test_filepath = "../../example_data/test_data.csv"
    # data = np.genfromtxt(bjd_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # bjd_data_no_occs = np.genfromtxt(bjd_no_occs_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # isot_data = np.genfromtxt(isot_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # jd_utc_data = np.genfromtxt(jd_utc_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # jd_utc_no_occs_data = np.genfromtxt(jd_utc_no_occs_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    test_data = np.genfromtxt(test_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # Set what data file you are using here
    data = test_data
    # STEP 2: Break data up into epochs, mid-times, and error
    # STEP 2.5 (Optional): Make sure the epochs are integers and not floats
    tra_or_occs = data["tra_or_occ"] # Base tra_or_occs
    epochs = data["epoch"].astype('int') # Epochs with tra_or_occs
    # epochs_no_occs = bjd_data_no_occs["epochs"]
    mid_times = data["mid_time"] # BJD mid times
    mid_time_errs = data["mid_time_err"] # BJD mid time errs
    print(f"epochs: {list(epochs)}")
    print(f"mid_times: {list(mid_times)}")
    print(f"mid_time_errs: {list(mid_time_errs)}")
    print(f"tra_or_occ: {list(tra_or_occs)}")
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
    # # STEP 4: Create new ephemeris object with transit times object
    ephemeris_obj1 = Ephemeris(times_obj1)
    # STEP 5: Get model ephemeris data & BIC values
    # # LINEAR MODEL
    # linear_model_data = ephemeris_obj1.get_model_ephemeris('linear')
    # print(linear_model_data)
    # linear_model_uncertainties = ephemeris_obj1.get_ephemeris_uncertainties(linear_model_data)
    # print(linear_model_uncertainties)
    # lin_bic = ephemeris_obj1.calc_bic(linear_model_data)
    # print(lin_bic)
    # QUADRATIC MODEL
    quad_model_data = ephemeris_obj1.get_model_ephemeris('quadratic')
    # print(quad_model_data)
    # quad_model_uncertainties = ephemeris_obj1.get_ephemeris_uncertainties(quad_model_data)
    # print(quad_model_uncertainties)
    # quad_bic = ephemeris_obj1.calc_bic(quad_model_data)
    # print(quad_bic)
    # STEP 5.5a: Get the delta BIC value for the linear and quadratic models
    # delta_bic_lq = ephemeris_obj1.calc_delta_bic("linear", "quadratic")
    # print(delta_bic_lq)
    # STEP 5.5b: Get the delta BIC value for the linear and precession models
    # delta_bic_lp = ephemeris_obj1.calc_delta_bic("linear", "precession")
    # print(delta_bic_lp)

    # PRECESSION MODEL
    # precession_model_data = ephemeris_obj1.get_model_ephemeris("precession")
    # print(precession_model_data)

    # STEP 6: Show a plot of the model ephemeris data
    # ephemeris_obj1.plot_model_ephemeris(linear_model_data, save_plot=False)
    # plt.show()
    # ephemeris_obj1.plot_model_ephemeris(quad_model_data, save_plot=False)
    # plt.show()

    # STEP 7: Uncertainties plot
    # ephemeris_obj1.plot_timing_uncertainties(linear_model_data, save_plot=False)
    # plt.show()
    # ephemeris_obj1.plot_timing_uncertainties(quad_model_data, save_plot=False)
    # plt.show()
    
    # STEP 8: O-C Plot
    # ephemeris_obj1.plot_oc_plot("quadratic", save_plot=False)
    # plt.show()

    # ephemeris_obj1._get_eclipse_duration("TrES-3 b")
    # observer_obj = ephemeris_obj1.create_observer_obj(timezone="US/Mountain", longitude=-116.208710, latitude=43.602,
    #                                                   elevation=821, name="BoiseState")
    # target_obj = ephemeris_obj1.create_target_obj("TrES-3")
    # obs_sch = ephemeris_obj1.get_observing_schedule(quad_model_data, "US/Mountain", observer_obj, target_obj, 10, 2, "2024-12-04", exoplanet_name="TrES-3")

    # TODO: Do timezone checks and stuff in observing sched 

    # STEP 9: Running delta BIC plot
    # running_bic_plot = ephemeris_obj1.plot_running_delta_bic(model1="linear", model2="quadratic", save_plot=False)
    # plt.show()

    # nea_data = ephemeris_obj1._get_eclipse_system_params("WASP-12 b", ra=None, dec=None)
    # # nea_data = ephemeris_obj1._query_nasa_exoplanet_archive("WASP-12 b", select_query="pl_ratror,pl_orbsmax,pl_imppar,pl_orbincl")
    # print(nea_data)
    # print(np.arcsin(0.3642601363) * 0.3474 * 24)