from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget, Observer, EclipsingSystem
# from susie.timing_data import TimingData # REMEMBER TO ONLY USE THIS FOR PACKAGE UPDATES
from .timing_data import TimingData # REMEMBER TO COMMENT THIS OUT BEFORE GIT PUSHES
# from timing_data import TimingData # REMEMBER TO COMMENT THIS OUT BEFORE GIT PUSHES

class BaseModelEphemeris(ABC):
    """Abstract class that defines the structure of different model ephemeris classes."""
    @abstractmethod
    def fit_model(self, x, y, yerr, tra_or_occ):
        """Fits a model ephemeris to timing data.

        Defines the structure for fitting a model (linear or quadratic) to timing data. 
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
                Indicates if each point of data is taken from a transit or an occultation.

        Returns
        ------- 
            A dictionary containing fitted model parameters. 
        """
        pass


class LinearModelEphemeris(BaseModelEphemeris):
    """Subclass of BaseModelEphemeris that implements a linear fit."""
    def lin_fit(self, E, P, T0, tra_or_occ):
        """Calculates a linear function with given data.

        Uses the equation 
         - (period * epochs + initial mid time) for transit observations 
         - (period * epochs + (initial mid time + (½ * period ))) for occultation observations 
        
        as a linear function for an LMfit Model.
        
        Parameters
        ----------
            E: numpy.ndarray[float]
                The epochs.
            P: float
                The exoplanet orbital period.
            T0: float
                The initial mid-time, also known as conjunction time.
            tra_or_occ: 
                Indicates if the data is from a transit or occultation.
        
        Returns
        -------
            result: numpy.ndarray[float]
                A linear function to be used with the LMfit Model, returned as:
                    :math:`P*E + T_0` if the data point is an observed transit (denoted by 0)
                    :math:`P*E + (T_0 + \\frac{1}{2}*P)` if the data point is an observed occultation (denoted by 1)
        """
        result = np.zeros_like(E)
        for i, t_type in enumerate(tra_or_occ):
            if t_type == 0:
                # transit data
                result[i] = P*E[i] + T0
            elif t_type == 1:
                # occultation data
                result[i] = P*E[i] + (T0 + 0.5*P)
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
        model = Model(self.lin_fit, independent_vars=['E', 'tra_or_occ'])
        # TODO: Should we set this as the base estimate for T0 and P or should we try to find a good estimate to start with?
        params = model.make_params(T0=0., P=1.091423, tra_or_occ=tra_or_occ_enum)
        result = model.fit(y, params, weights=1.0/yerr, E=x, tra_or_occ=tra_or_occ_enum)
        return_data = {
            'period': result.params['P'].value,
            'period_err': result.params['P'].stderr,
            'conjunction_time': result.params['T0'].value,
            'conjunction_time_err': result.params['T0'].stderr
        }
        return(return_data)

class QuadraticModelEphemeris(BaseModelEphemeris):
    """Subclass of BaseModelEphemeris that implements a quadratic fit."""
    def quad_fit(self, E, dPdE, P, T0, tra_or_occ):
        """Calculates a quadratic function with given data.

        Uses the equation 
         - ((0.5 * change in period over epoch * (epoch²)) + (period * epoch) + conjunction time) for transit observations
         - ((0.5 * change in period over epoch * (epoch²)) + (period * epoch) + conjunction time) for occultation observations as a quadratic function for the LMfit Model.
        
        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs.
            dPdE: float
                Change in period with respect to epoch.
            P: float
                The exoplanet orbital period.
            T0: float
                The initial mid-time, also known as conjunction time.
            tra_or_occ: 
                Indicates if the data is from a transit or occultation.
        
        Returns
        -------
            result: numpy.ndarray[float]
                A quadratic function to be used with the LMfit Model, returned as:
                    :math:`\\frac{1}{2}*\\frac{dP}{dE}*E^2 + P*E + T_0` if the data point is an observed transit (denoted by 0)
                    :math:`\\frac{1}{2}*\\frac{dP}{dE}*E^2 + P*E + (T_0 + \\frac{1}{2}*P)` if the data point is an observed occultation (denoted by 1)
        """
        result = np.zeros_like(E)
        for i, t_type in enumerate(tra_or_occ):
            if t_type == 0:
                # transit data
                result[i] = T0 + P*E[i] + 0.5*dPdE*E[i]*E[i] 
            elif t_type == 1:
                # occultation data
                result[i] = (T0 + 0.5*P) + P*E[i] + 0.5*dPdE*E[i]*E[i] 
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
        model = Model(self.quad_fit, independent_vars=['E', 'tra_or_occ'])
        # TODO: Should we set this as the base estimate for T0 and P or should we try to find a good estimate to start with?
        params = model.make_params(T0=0., P=1.091423, dPdE=0., tra_or_occ=tra_or_occ_enum)
        result = model.fit(y, params, weights=1.0/yerr, E=x, tra_or_occ=tra_or_occ_enum)
        return_data = {
            'period': result.params['P'].value,
            'period_err': result.params['P'].stderr,
            'conjunction_time': result.params['T0'].value,
            'conjunction_time_err': result.params['T0'].stderr,
            'period_change_by_epoch': result.params['dPdE'].value,
            'period_change_by_epoch_err': result.params['dPdE'].stderr
        }
        return(return_data)

class ModelEphemerisFactory:
    """Factory class for selecting which type of ephemeris class (linear or quadratic) to use."""
    @staticmethod
    def create_model(model_type, x, y, yerr, tra_or_occ):
        """Instantiates the appropriate BaseModelEphemeris subclass and runs fit_model method.

        Based on the given user input of model type (linear or quadratic) the factory will create the 
        corresponding subclass of BaseModelEphemeris and run the fit_model method to recieve the model 
        ephemeris return data dictionary.
        
        Parameters
        ----------
            model_type: str
                The name of the model ephemeris to create, either 'linear' or 'quadratic'.
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
        
        Raises
        ------
            ValueError:
                If model specified is not a valid subclass of BaseModelEphemeris, which is either 'linear' or 'quadratic'.
        """
        models = {
            'linear': LinearModelEphemeris(),
            'quadratic': QuadraticModelEphemeris()
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
                Either 'linear' or 'quadratic'. The ephemeris subclass specified to create and run.

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

        Raises
        ------
            ValueError:
                If model specified is not a valid subclass of BaseModelEphemeris, which is either 'linear' or 'quadratic'.
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
        else:
            raise ValueError('Only linear and quadratic models are supported at this time.')
    
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
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(np.sqrt((T0_err**2) + ((self.timing_data.epochs[i]**2)*(P_err**2))))
            elif t_type == 'occ':
                # occultation data
                result.append(np.sqrt((T0_err**2) + (((self.timing_data.epochs[i]+0.5)**2)*(P_err**2))))
        return np.array(result)
    
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
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(np.sqrt((T0_err**2) + ((self.timing_data.epochs[i]**2)*(P_err**2)) + ((1/4)*(self.timing_data.epochs[i]**4)*(dPdE_err**2))))
            elif t_type == 'occ':
                # occultation data
                result.append(np.sqrt((T0_err**2) + (((self.timing_data.epochs[i]+0.5)**2)*(P_err**2)) + ((1/4)*(self.timing_data.epochs[i]**4)*(dPdE_err**2))))
        return np.array(result)
    
    def _calc_linear_ephemeris(self, E, P, T0):
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
            P: float
                The orbital period of the exoplanet as calculated by the linear ephemeris model.
            T0: float
                The conjunction time of the exoplanet as calculated by the linear ephemeris model.

        Returns
        -------
            A numpy array of mid-time times calculated over each epoch using the equation above.
        """
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(T0 + (P*E[i]))
            elif t_type == 'occ':
                # occultation data
                result.append((T0 + 0.5*P) + (P*E[i]))
        return np.array(result)
    
    def _calc_quadratic_ephemeris(self, E, P, T0, dPdE):
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
            P: float
                The orbital period of the exoplanet as calculated by the linear ephemeris model.
            T0: float
                The conjunction time of the exoplanet as calculated by the linear ephemeris model.
            dPdE: float
                The period change with respect to epoch as calculated by the linear ephemeris model.

        Returns
        -------
            A numpy array of mid-times calculated over each epoch using the equation above.
        """
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(T0 + P*E[i] + 0.5*dPdE*E[i]*E[i])
            elif t_type == 'occ':
                # occultation data
                result.append((T0 + 0.5*P) + P*E[i] + 0.5*dPdE*E[i]*E[i])
        return np.array(result)
    
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
    
    def _subtract_plotting_parameters(self, model_mid_times, T0, P, E):
        """Subtracts the first terms to show smaller changes for plotting functions.

        Uses the equation:
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
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(model_mid_times[i] - T0 - (P*E[i]))
            elif t_type == 'occ':
                # occultation data
                result.append(model_mid_times[i] - T0 - (0.5*P) - (P*E[i]))
        return np.array(result)
    
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
        model_ephemeris_data['model_type'] = model_type
        # Once we get parameters back, we call _calc_linear_ephemeris 
        if model_type == 'linear':
            # Return dict with parameters and model data
            model_ephemeris_data['model_data'] = self._calc_linear_ephemeris(self.timing_data.epochs, model_ephemeris_data['period'], model_ephemeris_data['conjunction_time'])
        elif model_type == 'quadratic':
            model_ephemeris_data['model_data'] = self._calc_quadratic_ephemeris(self.timing_data.epochs, model_ephemeris_data['period'], model_ephemeris_data['conjunction_time'], model_ephemeris_data['period_change_by_epoch'])
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
        if 'model_type' not in model_params:
            raise KeyError("Cannot find model type in model data. Please run the get_model_ephemeris method to return ephemeris fit parameters.")
        if model_params['model_type'] == 'linear':
            if 'conjunction_time_err' not in model_params or 'period_err' not in model_params:
                raise KeyError("Cannot find conjunction time and period errors in model data. Please run the get_model_ephemeris method with 'linear' model_type to return ephemeris fit parameters.")
            return self._calc_linear_model_uncertainties(model_params['conjunction_time_err'], model_params['period_err'])
        elif model_params['model_type'] == 'quadratic':
            if 'conjunction_time_err' not in model_params or 'period_err' not in model_params or 'period_change_by_epoch_err' not in model_params:
                raise KeyError("Cannot find conjunction time, period, and/or period change by epoch errors in model data. Please run the get_model_ephemeris method with 'quadratic' model_type to return ephemeris fit parameters.")
            return self._calc_quadratic_model_uncertainties(model_params['conjunction_time_err'], model_params['period_err'], model_params['period_change_by_epoch_err'])
    
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

    def calc_delta_bic(self):
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

        Returns
        ------- 
            delta_bic : float
                Represents the :math:`\\Delta BIC` value for this timing data. 
        """
        linear_data = self.get_model_ephemeris('linear')
        quadratic_data = self.get_model_ephemeris('quadratic')
        linear_bic = self.calc_bic(linear_data)
        quadratic_bic = self.calc_bic(quadratic_data)
        delta_bic = linear_bic - quadratic_bic
        return delta_bic
    
    def _create_observer_obj(self, timezone, coords=None, name=None):
        """Creates the Astroplan Observer object.

        Parameters
        ----------
            timezone: str
                The local timezone. If a string, it will be passed through pytz.timezone() to produce the timezone object.
            coords: tuple(float, float, float) (Optional)
                The longitude, latitude, and elevation of the observer's location on Earth.
            name: str (Optional)
                The name of the observer's location. This can either be a registered Astropy site
                name (get the latest site names with `EarthLocation.get_site_names()`), which will
                return the latitude, longitude, and elevation of the site OR it can be a custom name
                to keep track of your Observer object.

        Returns
        -------
            The Astroplan Observer object.
        
        Raises
        ------
            ValueError if neither coords nor name are given.
        """
        observer = None
        if not any(val is None for val in coords):
            # There are valid vals for lon and lat
            observer = Observer(longitude=coords[0]*u.deg, latitude=coords[1]*u.deg, elevation=coords[2]*u.m, timezone=timezone)
            if name is not None:
                observer.name = name
        elif name is not None:
            # No vals for lon and lat, use site name
            observer = Observer.at_site(name, timezone=timezone)
        else:
            # No coords or site name given, raise error
            raise ValueError("Observatory location must be specified with either (1) a site name specified by astropy.coordinates.EarthLocation.get_site_names() or (2) latitude and longitude in degrees as accepted by astropy.coordinates.Latitude and astropy.coordinates.Latitude.")
        return observer
    
    def _create_target_obj(self, coords=None, name=None):
        """Creates the Astroplan FixedTarget object.

        Parameters
        ----------
            coords: tuple(float, float) (Optional)
                The right ascension and declination of the object in the sky (most likely the planet's host star).
            name: str (Optional)
                The name of the object in the sky. This can either be a registered object name, which will query
                a CDS name resolver (see Astroplan documentation for more information on this) OR it can be a 
                custom name to keep track of your FixedTarget object.

        Returns
        -------
            The Astroplan FixedTarget object.

        Raises
        ------
            ValueError if neither coords nor name are given.
        """
        target = None
        if not any(val is None for val in coords):
            # There are valid vals for ra and dec
            skycoord = SkyCoord(ra=coords[0]*u.deg, dec=coords[1]*u.deg)
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
    
    def get_observing_schedule(self, model_data_dict, timezone, obs_lat=None, obs_lon=None, obs_elevation=0.0, obs_name=None, obj_ra=None, obj_dec=None, obj_name=None, system_name=None):
        """Returns a list of observable future transits for the target object

        Parameters
        ----------
            model_data_dict: dict
                The model ephemeris data dictionary recieved from the `get_model_ephemeris` method.
            timezone: str
                The local timezone. If a string, it will be passed through `pytz.timezone()` to produce the timezone object.
            obs_lat: float (Optional)
                The latitude of the observer's location on Earth.
            obs_lon: float (Optional)
                The longitude of the observer's location on Earth.
            obs_elevation: float (Optional)
                The elevation of the observer's location on Earth.
            obs_name: str (Optional)
                The name of the observer's location. This can either be a registered Astropy site
                name (get the latest site names with `EarthLocation.get_site_names()`), which will
                return the latitude, longitude, and elevation of the site OR it can be a custom name
                to keep track of your `Observer` object.
            obj_ra: float (Optional)
                The right ascension of the object to observe in the sky (most likely a planet or star).
            obj_dec: float (Optional)
                The declination of the object to observe in the sky (most likely a planet or star).
            obj_name: str (Optional)
                The name of the object in the sky. This can either be a registered object name, which will query
                a CDS name resolver (see Astroplan documentation for more information on this) OR it can be a 
                custom name to keep track of your `FixedTarget` object.
            system_name: str (Optional)
                The name of your eclipsing system. An Optional parameter to help you keep track of your 
                `EclipsingSystem` object.
        """
        observer = self._create_observer_obj(timezone, coords=(obs_lon, obs_lat, obs_elevation), name=obs_name)
        target = self._create_target_obj(coords=(obj_ra, obj_dec), name=obj_name)
        # This is just a mid transit time, we most likely want to use the most recent mid transit time
        # For now just using last value from calculated mid-times 
        # TODO: Should we change this to use the most recent mid transit time from the data?
        primary_eclipse_time = Time(model_data_dict['model_data'][-1], format='jd')
        # We can pull orbital period from the model
        orbital_period = model_data_dict['period'] * u.day
        # TODO: Need to use the equation from the book for the eclipse duration
        eclipse_duration = 0.1277 * u.day
        eclipsing_system = EclipsingSystem(primary_eclipse_time=primary_eclipse_time,
                                orbital_period=orbital_period, duration=eclipse_duration,
                                name=system_name)
        pass
    
    def plot_model_ephemeris(self, model_data_dict, save_plot=False, save_filepath=None):
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
        plt.scatter(x=self.timing_data.epochs, y=model_data_dict['model_data'], color='#0033A0')
        plt.xlabel('Epochs')
        plt.ylabel('Model Predicted Mid-Times (units)')
        plt.title(f'Predicted {model_data_dict["model_type"].capitalize()} Model Mid Times over Epochs')
        if save_plot == True:
            plt.savefig(save_filepath)
        plt.show()

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
        model_uncertainties = self.get_ephemeris_uncertainties(model_data_dict)
        x = self.timing_data.epochs
        # get T(E) - T0 - PE  OR  T(E) - T0 - 0.5P - PE
        # TODO: Make this calculation a separate function
        y = self._subtract_plotting_parameters(model_data_dict['model_data'], model_data_dict['conjunction_time'], model_data_dict['period'], self.timing_data.epochs)
        # plot the y line, then the line +- the uncertainties
        plt.plot(x, y, c='blue', label='$t(E) - T_{0} - PE$')
        plt.plot(x, y + model_uncertainties, c='red', label='$(t(E) - T_{0} - PE) + σ_{t^{pred}_{tra}}$')
        plt.plot(x, y - model_uncertainties, c='red', label='$(t(E) - T_{0} - PE) - σ_{t^{pred}_{tra}}$')
        # Add labels and show legend
        plt.xlabel('Epochs')
        plt.ylabel('Seconds') # TODO: Are these days or seconds?
        plt.title(f'Uncertainties of Predicted {model_data_dict["model_type"].capitalize()} Model Ephemeris Mid Times')
        plt.legend()
        if save_plot is True:
            plt.savefig(save_filepath)
        plt.show()

    def plot_oc_plot(self, save_plot=False, save_filepath=None):
        """Plots a scatter plot of observed minus calculated values of mid-times for linear and quadratic model ephemerides over epochs.

        Parameters
        ----------
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        # y = T0 - PE - 0.5 dP/dE E^2
        lin_model = self.get_model_ephemeris('linear')
        quad_model = self.get_model_ephemeris('quadratic')
        # y = 0.5 dP/dE * (E - median E)^2
        # TODO: Make this calculation a separate function
        quad_model_curve = ((1/2)*quad_model['period_change_by_epoch'])*((self.timing_data.epochs - np.median(self.timing_data.epochs))**2)
        # plot points w/ x=epoch, y=T(E)-T0-PE, yerr=sigmaT0
        y = self._subtract_plotting_parameters(self.timing_data.mid_times, lin_model['conjunction_time'], lin_model['period'], self.timing_data.epochs)
        plt.errorbar(self.timing_data.epochs, y, yerr=self.timing_data.mid_time_uncertainties, 
                    marker='o', ls='', color='#0033A0',
                    label=r'$t(E) - T_0 - P E$')
        plt.plot(self.timing_data.epochs,
                 (quad_model_curve),
                 color='#D64309', label=r'$\frac{1}{2}(\frac{dP}{dE})E^2$')
        plt.legend()
        plt.xlabel('E - Median E')
        plt.ylabel('O-C (seconds)')
        plt.title('Observed Minus Caluclated Plot')
        if save_plot is True:
            plt.savefig(save_filepath)
        plt.show()

    def plot_running_delta_bic(self, save_plot=False, save_filepath=None):
        """Plots a scatterplot of epochs vs. :math:`\\Delta BIC` for each epoch.

        Starting at the third epoch, will plot the value of :math:`\\Delta BIC` for all previous epochs,
        showing how the value of :math:`\\Delta BIC` progresses over time with more observations.

        Parameters
        ----------
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        delta_bics = []
        all_epochs = self.timing_data.epochs
        all_mid_times = self.timing_data.mid_times
        all_uncertainties = self.timing_data.mid_time_uncertainties
        all_tra_or_occ = self.timing_data.tra_or_occ
        # for each epoch (starting at 3?), calculate the delta bic, plot delta bics over epoch
        for i in range(0, len(all_epochs)):
            if i < 2:
                delta_bics.append(int(0))
            else:
                self.timing_data.epochs = all_epochs[:i+1]
                self.timing_data.mid_times = all_mid_times[:i+1]
                self.timing_data.mid_time_uncertainties = all_uncertainties[:i+1]
                self.timing_data.tra_or_occ = all_tra_or_occ[:i+1]
                delta_bic = self.calc_delta_bic()
                delta_bics.append(delta_bic)
        plt.scatter(x=self.timing_data.epochs, y=delta_bics, color='#0033A0')
        plt.grid(True)
        plt.plot(self.timing_data.epochs, delta_bics, color='#0033A0')
        plt.xlabel('Epoch')
        plt.ylabel('$\Delta$BIC')
        plt.title("Value of $\Delta$BIC as Observational Epochs Increase")
        if save_plot is True:
            plt.savefig(save_filepath)
        plt.show()

if __name__ == '__main__':
    # STEP 1: Upload datra from file
    filepath = "../../example_data/wasp12b_tra_occ.csv"
    # filepath = "../../malia_examples/WASP12b_transit_ephemeris.csv"
    data = np.genfromtxt(filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # STEP 2: Break data up into epochs, mid-times, and error
    # STEP 2.5 (Optional): Make sure the epochs are integers and not floats
    tra_or_occs = data["tra_or_occ"]
    epochs = data["epoch"].astype('int')
    mid_times = data["transit_time"]
    mid_time_errs = data["sigma_transit_time"]
    # print(f"epochs: {list(epochs)}")
    # print(f"mid_times: {list(mid_times)}")
    # print(f"mid_time_errs: {list(mid_time_errs)}")
    # print(f"tra_or_occ: {list(tra_or_occs)}")
    # STEP 3: Create new transit times object with above data
    # times_obj1 = TimingData('jd', epochs, mid_times, mid_time_errs, tra_or_occ=tra_or_occs, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    # times_obj1 = TimingData('jd', epochs, mid_times, mid_time_errs, time_scale='tdb')
    times_obj1 = TimingData('jd', epochs, mid_times, mid_time_errs, time_scale='tdb', tra_or_occ=tra_or_occs)
    print(vars(times_obj1))
    # STEP 4: Create new ephemeris object with transit times object
    ephemeris_obj1 = Ephemeris(times_obj1)
    # STEP 5: Get model ephemeris data & BIC values
    # # LINEAR MODEL
    linear_model_data = ephemeris_obj1.get_model_ephemeris('linear')
    print(linear_model_data)
    linear_model_uncertainties = ephemeris_obj1.get_ephemeris_uncertainties(linear_model_data)
    # print(linear_model_uncertainties)
    lin_bic = ephemeris_obj1.calc_bic(linear_model_data)
    # print(lin_bic)
    # # QUADRATIC MODEL
    quad_model_data = ephemeris_obj1.get_model_ephemeris('quadratic')
    # print(quad_model_data)
    quad_model_uncertainties = ephemeris_obj1.get_ephemeris_uncertainties(quad_model_data)
    # print(quad_model_uncertainties)
    quad_bic = ephemeris_obj1.calc_bic(quad_model_data)
    # print(quad_bic)
    # STEP 5.5: Get the delta BIC value for both models
    delta_bic = ephemeris_obj1.calc_delta_bic()
    # print(delta_bic)

    # STEP 6: Show a plot of the model ephemeris data
    # ephemeris_obj1.plot_model_ephemeris(linear_model_data, save_plot=False)
    # ephemeris_obj1.plot_model_ephemeris(quad_model_data, save_plot=False)

    # STEP 7: Uncertainties plot
    # ephemeris_obj1.plot_timing_uncertainties(linear_model_data, save_plot=False)
    # ephemeris_obj1.plot_timing_uncertainties(quad_model_data, save_plot=False)
    
    # STEP 8: O-C Plot
    ephemeris_obj1.plot_oc_plot(save_plot=False)

    # STEP 9: Running delta BIC plot
    # ephemeris_obj1.plot_running_delta_bic(save_plot=False)
    
