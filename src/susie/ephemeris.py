from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
# from susie.timing_data import TimingData # REMEMBER TO ONLY USE THIS FOR PACKAGE UPDATES
from .timing_data import TimingData # REMEMBER TO COMMENT THIS OUT BEFORE GIT PUSHES

class BaseModelEphemeris(ABC):
    """Abstract class that defines the structure of different model ephemeris classes."""
    @abstractmethod
    def fit_model(self, x, y, yerr, tra_or_occ, **kwargs):
        """Fits a model ephemeris to transit data.

        TODO: Update docstring
        Defines the structure for fitting a model (linear or quadratic) to transit data. 
        All subclasses must implement this method.

        Parameters
        ----------
            x : numpy.ndarray[int]
                The epoch data as recieved from the TransitTimes object.
            y : numpy.ndarray[float]
                The mid transit time data as recieved from the TransitTimes object.
            yerr : numpy.ndarray[float]
                The mid transit time error data as recieved from the TransitTimes object.

        Returns
        ------- 
            A dictionary containing fitted model parameters. 

        """
        pass


class LinearModelEphemeris(BaseModelEphemeris):
    """Subclass of BaseModelEphemeris that implements a linear fit."""
    def lin_fit(self, E, P, T0, tra_or_occ):
        """Calculates a linear function with given data.
        TODO: Update docstring
        Uses the equation (Period * epochs + initial mid time) as a linear function for SciPy's 
        curve_fit method.
        
        Parameters
        ----------

            E: numpy.ndarray[float]
                The epochs.
            P: float
                The exoplanet transit period.
            T0: float
                The initial mid time.
        
        Returns
        -------
            result: numpy.ndarray[float]
                A linear function calculated with the TimingData object, returned as:
                    P*E + T0 if the data point is an observed transit (denoted by 0)
                    P*E + (T0 + 0.5*P) if the data point is an observed occultation (denoted by 1)
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
    
    def fit_model(self, x, y, yerr, tra_or_occ, **kwargs):
        """Fits a linear model to ephemeris data.

        TODO: Update docstring
        Compares the model ephemieris data to the linear fit created by data in TransitTimes object calculated 
        with lin_fit method. Then creates a curve fit which minimizes the difference between the two sets of data.
        Curve fit then returns the parameters of the linear function corresponding to period, conjunction time, 
        and their respective errors. These parameters are returned in a dictionary to the user for further use.

        Parameters
        ----------
            x: numpy.ndarray[int]
                The epoch data as recieved from the TransitTimes object.
            y: numpy.ndarray[float]
                The mid transit time data as recieved from the TransitTimes object.
            yerr: numpy.ndarray[float]
                The mid transit time error data as recieved from the TransitTimes object.
            **kwargs:
                Any key word arguments to be used in the scipy.optimize.curve_fit method.

        Returns
        ------- 
        return_data: dict
            A dictionary of parameters from the fit model ephemeris. 
            
            Example:
                * 'period': An array of exoplanet periods over time corresponding to epochs (in units of days),
                * 'period_err': The uncertainities associated with period (in units of days),
                * 'conjunction_time': The time of conjunction of exoplanet transit over time corresponding to epochs,
                * 'conjunction_time_err': The uncertainties associated with conjunction_time
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

        TODO: Update docstring
        Uses the equation (0.5 * change in period over epoch * mid transit times^2 + Period * mid transit times + initial epoch) 
        as a quadratic function for SciPy's curve_fit method.
        
        Parameters
        ----------
            x: numpy.ndarray[int]
                The mid-transit times.
            dPdE: float
                Change in period with respect to epoch.
            P: float
                The exoplanet transit period.
            T0: float
                The initial epoch associated with a mid-transit time.
        
        Returns
        -------
            0.5*dPdE*x*x + P*x + T0: numpy.ndarray[float]
                A list of quadratic function values calculated with TransitTimes object data to be 
                used with curve_fit.
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
    
    def fit_model(self, x, y, yerr, tra_or_occ, **kwargs):
        """Fits a quadratic model to ephemeris data.

        TODO: Update docstring
        Compares the model ephemeris data to the quadratic fit calculated with quad_fit method. Then creates a 
        curve fit which minimizes the difference between the two sets of data. Curve fit then returns the 
        parameters of the quadratic function corresponding to period, conjunction time, period change by epoch, 
        and their respective errors. These parameters are returned in a dictionary to the user for further use.

        Parameters
        ----------
            x: numpy.ndarray[int]
                The epoch data as recieved from the TransitTimes object.
            y: numpy.ndarray[float]
                The mid transit time data as recieved from the TransitTimes object.
            yerr: numpy.ndarray[float]
                The mid transit time error data as recieved from the TransitTimes object.
            **kwargs:
                Any key word arguments to be used in the scipy.optimize.curve_fit method.

        Returns
        ------- 
        return_data: dict
            A dictionary of parameters from the fit model ephemeris. Example:
                * 'period': An array of exoplanet periods over time corresponding to epochs (in units of days),
                * 'period_err': The uncertainities associated with period (in units of days),
                * 'conjunction_time': The time of conjunction of exoplanet transit over time corresponding to epochs,
                * 'conjunction_time_err': The uncertainties associated with conjunction_time,
                * 'period_change_by_epoch': The exoplanet period change over epochs, from first epoch to current epoch (in units of days),
                * 'period_change_by_epoch_err': The uncertainties associated with period_change_by_epoch (in units of days)
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
    def create_model(model_type, x, y, yerr, tra_or_occ, **kwargs):
        """Instantiates the appropriate BaseModelEphemeris subclass and runs fit_model method.

        TODO: Update docstring
        Based on the given user input of model type (linear or quadratic) the factory will create the 
        corresponding subclass of BaseModelEphemeris and run the fit_model method to recieve the model 
        ephemeris return data dictionary.
        
        Parameters
        ----------
            model_type: str
                The name of the model ephemeris to create, either 'linear' or 'quadratic'.
            x: numpy.ndarray[int]
                The epoch data as recieved from the TransitTimes object.
            y: numpy.ndarray[float]
                The mid transit time data as recieved from the TransitTimes object.
            yerr: numpy.ndarray[float]
                The mid transit time error data as recieved from the TransitTimes object.
            **kwargs:
                Any keyword arguments to be used in the scipy.optimize.curve_fit method.

        Returns
        ------- 
            Model : dict
                A dictionary of parameters from the fit model ephemeris. If a linear model was chosen, these parameters are:
                    * 'period': An array of exoplanet periods over time corresponding to epochs (in units of days),
                    * 'period_err': The uncertainities associated with period (in units of days),
                    * 'conjunction_time': The time of conjunction of exoplanet transit over time corresponding to epochs,
                    * 'conjunction_time_err': The uncertainties associated with conjunction_time
                If a quadratic model was chosen, the same variables are returned, and an additional parameter is included in the dictionary:
                    * 'period_change_by_epoch': The exoplanet period change over epochs, from first epoch to current epoch (in units of days),
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
        return model.fit_model(x, y, yerr, tra_or_occ, **kwargs)


class Ephemeris(object):
    """Represents the model ephemeris using transit midpoint data over epochs.
    TODO: Update docstring
    Parameters
    -----------
    transit_times: TransitTimes obj
        A successfully instantiated TransitTimes object holding epochs, mid transit times, and uncertainties.
        
    Raises
    ----------
     ValueError:
        Raised if transit_times is not an instance of the TransitTimes object.
    """
    def __init__(self, timing_data):
        """Initializing the transit times object and model ephermeris object
        TODO: Update docstring
        Parameters
        -----------
        transit_times: TransitTimes obj
            A successfully instantiated TransitTimes object holding epochs, mid transit times, and uncertainties.
        
        Raises
        ------
            ValueError :
                error raised if 'transit_times' is not an instance of 'TransitTimes' object.
        """
        self.timing_data = timing_data
        self._validate()

    def _validate(self):
        """Check that transit_times is an instance of the TransitTimes object.
        TODO: Update docstring
        Raises
        ------
            ValueError :
                error raised if 'transit_times' is not an instance of 'TransitTimes' object.
        """
        if not isinstance(self.timing_data, TimingData):
            raise ValueError("Variable 'timing_data' expected type of object 'TimingData'.")
        
    def _get_timing_data(self):
        """Returns transit time data for use.

        TODO: Update docstring
        Returns the epoch, mid transit time, and mid transit time error data from the TransitTimes object.

        Returns
        -------
            x: numpy.ndarray[int]
                The epoch data as recieved from the TransitTimes object.
            y: numpy.ndarray[float]
                The mid transit time data as recieved from the TransitTimes object.
            yerr: numpy.ndarray[float]
                The mid transit time error data as recieved from the TransitTimes object.
        """
        x = self.timing_data.epochs
        y = self.timing_data.mid_times
        yerr = self.timing_data.mid_time_uncertainties
        tra_or_occ = self.timing_data.tra_or_occ
        return x, y, yerr, tra_or_occ
    
    def _get_model_parameters(self, model_type, **kwargs):
        """Creates the model ephemeris object and returns model parameters.
        
        TODO: Update docstring
        This method processes and fetches data from the TransitTimes object to be used in the model ephemeris. 
        It creates the appropriate subclass of BaseModelEphemeris using the ModelEphemeris factory, then runs 
        the fit_model method to return the model parameters dictionary.

        Parameters
        ----------
            model_type: str
                Either 'linear' or 'quadratic'. The ephemeris subclass specified to create and run.

        Returns
        -------
            model_ephemeris_data: dict
                A dictionary of parameters from the fit model ephemeris. If a linear model was chosen, these parameters are:
                {
                    'period': An array of exoplanet periods over time corresponding to epochs,
                    'period_err': The uncertainities associated with period,
                    'conjunction_time': The time of conjunction of exoplanet transit over time corresponding to epochs,
                    'conjunction_time_err': The uncertainties associated with conjunction_time
                }
                If a quadratic model was chosen, the same variables are returned, and an additional parameter is included in the dictionary:
                {
                    'period_change_by_epoch': The exoplanet period change over epochs, from first epoch to current epoch,
                    'period_change_by_epoch_err': The uncertainties associated with period_change_by_epoch,
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
            return ValueError('Only linear and quadratic models are supported at this time.')
    
    def _calc_linear_model_uncertainties(self, T0_err, P_err):
        """Calculates the uncertainties of a given linear model when compared to actual data in TransitTimes.
        
        TODO: Update docstring
        Uses the equation σ(t pred, tra) = √(σ(T0)^2 + σ(P)^2 * E^2) where σ(T0)=conjunction time error, 
        E=epoch, and σ(P)=period error, to calculate the uncertainties between the model data and actual 
        data over epochs.
        
        Parameters
        ----------
        T0_err: numpy.ndarray[float]
            The calculated conjunction time errors from a linear model ephemeris.
        P_err: numpy.ndarray[float]
            The calculated period errors from a linear model ephemeris.
        
        Returns
        -------
            A list of uncertainties associated with the model ephemeris data passed in, calculated with the 
            equation above and the TransitTimes epochs.
        """
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(np.sqrt((T0_err**2) + ((self.timing_data.epochs[i]**2)*(P_err**2))))
            elif t_type == 'occ':
                # occultation data
                result.append(np.sqrt(((T0_err**2) + (0.5 * (P_err**2))) + ((self.timing_data.epochs[i]**2)*(P_err**2))))
        return np.array(result)
    
    def _calc_quadratic_model_uncertainties(self, T0_err, P_err, dPdE_err):
        """Calculates the uncertainties of a given quadratic model when compared to actual data in TransitTimes.
        
        TODO: Update docstring
        Uses the equation σ(t pred, tra) = √(σ(T0)^2 + (σ(P)^2 * E^2) + (1/4 * σ(dP/dE)^2 * E^4)) where 
        σ(T0)=conjunction time error, E=epoch, σ(P)=period error, and σ(dP/dE)=period change by epoch error, 
        to calculate the uncertainties between the model data and actual data over epochs.
        
        Parameters
        ----------
        T0_err: numpy.ndarray[float]
            The calculated conjunction time errors from a quadratic model ephemeris.
        P_err: numpy.ndarray[float]
            The calculated period errors from a quadratic model ephemeris.
        dPdE_err: numpy.ndarray[float]
            The calculated change in epoch over period error for a quadratic model ephemeris.
        
        Returns
        -------
            A list of uncertainties associated with the model ephemeris passed in, calculated with the 
            equation above and the TransitTimes epochs.
        """
        # Have both transits and occultations, calculate different value for each
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(np.sqrt((T0_err**2) + ((self.timing_data.epochs[i]**2)*(P_err**2)) + ((1/4)*(self.timing_data.epochs[i]**4)*(dPdE_err**2))))
            elif t_type == 'occ':
                # occultation data
                result.append(np.sqrt(((T0_err**2) + (0.5 * (P_err**2))) + ((self.timing_data.epochs[i]**2)*(P_err**2)) + ((1/4)*(self.timing_data.epochs[i]**4)*(dPdE_err**2))))
        return np.array(result)
    
    def _calc_linear_ephemeris(self, E, P, T0):
        """Calculates the mid transit times using parameters from a linear model ephemeris.
        
        TODO: Update docstring
        Uses the equation (T0 + PE) to calculate the mid transit times over each epoch where T0 is 
        conjunction time, P is period, and E is epoch.

        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs pulled from the TransitTimes object.
            P: float
                The period of the exoplanet transit as calculated by the linear ephemeris model.
            T0: float
                The conjunction time of the exoplanet transit as calculated by the linear ephemeris model.

        Returns
        -------
            A numpy array of mid transit times calculated over each epoch using the equation above.
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
        """Calculates the mid transit times using parameters from a quadratic model ephemeris.

        TODO: Update docstring
        Uses the equation (T0 + PE + 0.5 * dPdE * E^2) to calculate the mid transit times over each epoch 
        where T0 is conjunction time, P is period, E is epoch, and dPdE is period change with respect to epoch.

        Parameters
        ----------
            E: numpy.ndarray[int]
                The epochs pulled from the TransitTimes object.
            P: float
                The period of the exoplanet transit as calculated by the linear ephemeris model.
            T0: float
                The conjunction time of the exoplanet transit as calculated by the linear ephemeris model.
            dPdE: float
                The period change with respect to epoch as calculated by the linear ephemeris model.

        Returns
        -------
            A numpy array of mid transit times calculated over each epoch using the equation above.
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
    
    def _calc_chi_squared(self, model_data):
        """Calculates the residual chi squared values for the model ephemeris.

        TODO: Update docstring
        STEP 1: Get the observed transit times and observed transit times uncertainties from transit_times.py.

        STEP 2: Calculate the chi-squared value for the observed and model data, then return this value.
        
        Parameters
        ----------
            model_data : numpy.ndarray[float]
                The 'model_data' values from the returned dictionary of fit model ephemeris method, representing the \\
                    predicted mid-transit time data, the inital period, and the conjunction time.
        
        Returns
        -------
            Chi-squared value : float
                The chi-squared value calculated from the observed and model data.
        """
        # STEP 1: Get observed transit times
        observed_data = self.timing_data.mid_times
        uncertainties = self.timing_data.mid_time_uncertainties
        # STEP 2: calculate X2 with observed data and model data
        return np.sum(((observed_data - model_data)/uncertainties)**2)
    
    def _subtract_plotting_parameters(self, model_data, T0, P, E):
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(model_data[i] - T0 - (P*E[i]))
            elif t_type == 'occ':
                # occultation data
                result.append(model_data[i] - T0 - (0.5*P) - (P*E[i]))
        return np.array(result)
    
    def get_model_ephemeris(self, model_type):
        """Fits the transit data to a specified model using scipy.optimize.curve_fit function.
        TODO: Update docstring
        Parameters
        ----------
            model_type: str
                Either 'linear' or 'quadratic'. Represents the type of ephemeris to fit the data to.

        Returns
        ------- 
            A dictionary of parameters from the fit model ephemeris. If a linear model was chosen, these parameters are:
            
                * 'period': An array of exoplanet periods over time corresponding to epochs (in units of days),
                * 'period_err': The uncertainities associated with period (in units of days),
                * 'conjunction_time': The time of conjunction of exoplanet transit over time corresponding to epochs,
                * 'conjunction_time_err': The uncertainties associated with conjunction_time
            
            If a quadratic model was chosen, the same variables are returned, and an additional parameter is included in the dictionary:
            
                * 'period_change_by_epoch': The exoplanet period change over epochs, from first epoch to current epoch (in units of days),
                * 'period_change_by_epoch_err': The uncertainties associated with period_change_by_epoch (in units of days),
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
        """Calculates the uncertainties of a specific model data when compared to the actual data. 
        TODO: Update docstring
        Uses the equation 
        
        .. math::
            \\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * E^2)}
        
        for linear models and 

        .. math::
            \\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * E^2) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))} 
        
        for quadratic models (where :math:`\\sigma(T_0) =` conjunction time error, :math:`E=` epoch, :math:`\\sigma(P)=` period error, and :math:`\\sigma(\\frac{dP}{dE})=` period change by epoch error) to calculate the uncertainties between the model data and actual data over epochs.
        
        Parameters
        ----------
        model_params: dict
            A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
        
        Returns
        -------
            A list of uncertainties associated with the model ephemeris passed in, calculated with the 
            equation above and the TransitTimes epochs.
        
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
        """
        Calculates the BIC value for a given model ephemeris. 
        
        Uses the equation

        .. math::
            BIC = \\chi^2 + (k * log(N))
         
        where :math:`\\chi^2=\\sum{  \\frac{(\\text{observed mid transit times - model mid transit times})}{\\text{(observed mid transit time uncertainties})^2}   },`  k=number of fit parameters (2 for linear models, 3 for quadratic models), and N=total number of data points.
        
        Parameters
        ----------
            model_data_dict: dict
                A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
        
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
        """Calculates the :math:`\\Delta BIC` value between linear and quadratic model ephemerides using the given transit data. 
        
        STEP 1: Calls get_model_ephemeris for both the linear and quadratic models. 

        STEP 2: Calls calc_bic for both the linear and quadratic sets of data.

        STEP 3: Calculates and returns :math:`\\Delta BIC,` which is the difference between the linear BIC and quadratic BIC.

        Returns
        ------- 
            delta_bic : float
                Represents the :math:`\\Delta BIC` value for this transit data. 
        """
        linear_data = self.get_model_ephemeris('linear')
        quadratic_data = self.get_model_ephemeris('quadratic')
        linear_bic = self.calc_bic(linear_data)
        quadratic_bic = self.calc_bic(quadratic_data)
        delta_bic = linear_bic - quadratic_bic
        return delta_bic
    
    def plot_model_ephemeris(self, model_data_dict, save_plot=False, save_filepath=None):
        """Returns a MatplotLib scatter plot showing predicted mid transit times from the model ephemeris over epochs.

        TODO: Update docstring
        STEP 1: Plot a scatterplot of epochs (from transit_times.py) vs model_data, which is an array of floats that is a value of 'model_data_dict'.

        STEP 2: Save the plot if indicated by the user.

        Parameters
        ----------
            model_data_dict: dict
                A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        
        Returns
        ------- 
            A MatplotLib plot of epochs vs. model predicted mid-transit times.
        """
        plt.scatter(x=self.timing_data.epochs, y=model_data_dict['model_data'])
        plt.xlabel('Epochs')
        plt.ylabel('Model Predicted Mid-Times (units)')
        plt.title(f'Predicted {model_data_dict["model_type"].capitalize()} Model Mid Times over Epochs')
        if save_plot == True:
            plt.savefig(save_filepath)
        plt.show()

    def plot_timing_uncertainties(self, model_data_dict, save_plot=False, save_filepath=None):
        """Returns a MatplotLib scatter plot showing timing uncertainties over epochs.
        TODO: Update docstring, hwo do we label the lines now that we have additional way of calculating y for occultations?
        STEP 1: Get the uncertianies from the model data dictionary.

        STEP 2: Get the model data, which is an arrary of floats representing the predicted mid-transit time data, the conjunction time and the inital period. Subtract the conjunction time and the initial period from this array.

        STEP 3: Plot this modified model data, showing the maximum and minimum model uncertainity at each point.

        STEP 4: Save the plot if indicated by the user. 

        Parameters
        ----------
            model_data_dict: dict
                A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        
        Returns
        ------- 
            A MatplotLib plot of timing uncertainties.
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
        """Returns a MatplotLib scatter plot showing observed vs. calculated values of mid transit times for linear and quadratic model ephemerides over epochs.

        TODO: Update docstring
        STEP 1: Call 'get_model_ephemeris' for both the linear and quadratic model types. 

        STEP 2: Calculate the quadratic model curve, which follows the formula :math:`y = 0.5 \\frac{dP_0}{dE} * (E - \\text{median} E)^2.`

        STEP 3: Plot the quadratic model curve vs. epochs from transit_times.py. Plot the error bars at each data point using the \\
        'mid_transit_times_uncertainties' from transit_times.py.

        STEP 4: Save the plot if indicated by the user. 

        Parameters
        ----------
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        
        Returns
        -------
            A MatplotLib plot of observed vs. calculated values of mid transit times for linear and quadratic model ephemerides over epochs.
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
        # TODO: Need to change this, how do we plot with occultations introduced? 
        # Maybe we still keep all epochs as a variable?
        # We also use mid times and uncertainties, so we will have to figure out how to account 
        # for the occultation data in here

        """Returns a MatPlotlib scatterplot of epochs vs. :math:`\\Delta BIC` for each epoch.

        STEP 1: Get the epochs, mid transit times and mid transit times uncertainties from 'transit_times.py'.

        STEP 2: Create a list of the :math:`\\Delta BIC` values. For the first 3 epochs, the :math:`\\Delta BIC` value is zero. For the subsequent epochs\\
        call 'calc_delta_bic' and append the returned value to the list of delta bic values.

        STEP 3: Plot a scatterplot of the epochs vs. the :math:`\\Delta BIC` values.

        STEP 4: Save the plot if indicated by the user. 

        Parameters
        ----------
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
                
        Returns
        -------
            A MatplotLib scatter plot of epochs vs. :math:`\\Delta BIC` for each epoch.
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
        plt.scatter(x=self.timing_data.epochs, y=delta_bics)
        plt.grid(True)
        plt.plot(self.timing_data.epochs, delta_bics)
        plt.xlabel('Epoch')
        plt.ylabel('$\Delta$BIC')
        plt.title("Value of $\Delta$BIC as Observational Epochs Increase")
        if save_plot is True:
            plt.savefig(save_filepath)
        plt.show()

if __name__ == '__main__':
    # STEP 1: Upload datra from file
    filepath = "../../wasp12b_tra_occ.csv"
    # filepath = "../../malia_examples/WASP12b_transit_ephemeris.csv"
    data = np.genfromtxt(filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # STEP 2: Break data up into epochs, mid transit times, and error
    # STEP 2.5 (Optional): Make sure the epochs are integers and not floats
    tra_or_occs = data["tra_or_occ"]
    epochs = data["epoch"].astype('int')
    mid_transit_times = data["transit_time"]
    mid_transit_times_err = data["sigma_transit_time"]
    # STEP 3: Create new transit times object with above data
    # times_obj1 = TimingData('jd', epochs, mid_transit_times, mid_transit_times_err, tra_or_occ=tra_or_occs, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    # times_obj1 = TimingData('jd', epochs, mid_transit_times, mid_transit_times_err, time_scale='tdb')
    times_obj1 = TimingData('jd', epochs, mid_transit_times, mid_transit_times_err, time_scale='tdb', tra_or_occ=tra_or_occs)
    # STEP 4: Create new ephemeris object with transit times object
    ephemeris_obj1 = Ephemeris(times_obj1)
    # STEP 5: Get model ephemeris data & BIC values
    # # LINEAR MODEL
    linear_model_data = ephemeris_obj1.get_model_ephemeris('linear')
    # print(linear_model_data)
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
    # ephemeris_obj1.plot_oc_plot(save_plot=False)

    # STEP 9: Running delta BIC plot
    ephemeris_obj1.plot_running_delta_bic(save_plot=False)
    
