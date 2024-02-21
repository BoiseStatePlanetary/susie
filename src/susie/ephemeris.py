from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
# from transit_times import TransitTimes
from susie.transit_times import TransitTimes

class BaseModelEphemeris(ABC):
    """Abstract class that defines the structure of different model ephemeris classes."""
    @abstractmethod
    def fit_model(self, x, y, yerr, **kwargs):
        """Fits a model ephemeris to transit data.

        Defines the structure for fitting a model (linear or quadratic) to transit data. 
        All subclasses must implement this method.

        Parameters
        ----------
            x: numpy.ndarray[int]
                The epoch data as recieved from the TransitTimes object.
            y: numpy.ndarray[float]
                The mid transit time data as recieved from the TransitTimes object.
            yerr: numpy.ndarray[float]
                The mid transit time error data as recieved from the TransitTimes object.

        Returns
        ------- 
            A dictionary containing fitted model parameters.
        """
        pass


class LinearModelEphemeris(BaseModelEphemeris):
    """Subclass of BaseModelEphemeris that implements a linear fit."""
    def lin_fit(self, x, P, T0):
        """Calculates a linear function with given data.

        Uses the equation (Period * mid transit times + initial epoch) as a linear function for SciPy's 
        curve_fit method.
        
        Parameters
        ----------
            x: numpy.ndarray[int]
                The mid-transit times.
            P: float
                The exoplanet transit period.
            T0: float
                The initial epoch associated with a mid-transit time.
        
        Returns
        -------
            P*x + T0: numpy.ndarray[float]
                A linear function calculated with TransitTimes object data to be used with curve_fit.
        """
        return P*x + T0
    
    def fit_model(self, x, y, yerr, **kwargs):
        """Fits a linear model to ephemeris data.

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
            A dictionary of parameters from the fit model ephemeris. Example:
                {
                    'period': An array of exoplanet periods over time corresponding to epochs,
                    'period_err': The uncertainities associated with period,
                    'conjunction_time': The time of conjunction of exoplanet transit over time corresponding to epochs,
                    'conjunction_time_err': The uncertainties associated with conjunction_time
                }
        """
        # Will come back in units of y/x (so will be in days), period is days per orbit (but rly this is days bc orbit is unitless)
        popt, pcov = curve_fit(self.lin_fit, x, y, sigma=yerr, absolute_sigma=True, **kwargs)
        unc = np.sqrt(np.diag(pcov))
        return_data = {
            'period': popt[0],
            'period_err': unc[0],
            'conjunction_time': popt[1],
            'conjunction_time_err': unc[1]
        }
        return(return_data)

# TODO: Check the units for each thing in here (are times given in days, seconds?)

class QuadraticModelEphemeris(BaseModelEphemeris):
    """Subclass of BaseModelEphemeris that implements a quadratic fit."""
    def quad_fit(self, x, dPdE, P, T0):
        """Calculates a quadratic function with given data.

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
        return 0.5*dPdE*x*x + P*x + T0
    
    def fit_model(self, x, y, yerr, **kwargs):
        """Fits a quadratic model to ephemeris data.

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
                {
                    'period': An array of exoplanet periods over time corresponding to epochs,
                    'period_err': The uncertainities associated with period,
                    'conjunction_time': The time of conjunction of exoplanet transit over time corresponding to epochs,
                    'conjunction_time_err': The uncertainties associated with conjunction_time,
                    'period_change_by_epoch': The exoplanet period change over epochs, from first epoch to current epoch,
                    'period_change_by_epoch_err': The uncertainties associated with period_change_by_epoch
                }
        """
        popt, pcov = curve_fit(self.quad_fit, x, y, sigma=yerr, absolute_sigma=True, **kwargs)
        unc = np.sqrt(np.diag(pcov))
        return_data = {
            'conjunction_time': popt[2],
            'conjunction_time_err': unc[2],
            'period': popt[1],
            'period_err': unc[1],
            'period_change_by_epoch': popt[0],
            'period_change_by_epoch_err': unc[0],
        }
        print(return_data)
        return(return_data)

class ModelEphemerisFactory:
    """Factory class for selecting which type of ephemeris class (linear or quadratic) to use."""
    @staticmethod
    def create_model(model_type, x, y, yerr, **kwargs):
        """Instantiates the appropriate BaseModelEphemeris subclass and runs fit_model method.

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
        models = {
            'linear': LinearModelEphemeris(),
            'quadratic': QuadraticModelEphemeris()
        }
        if model_type not in models:
            raise ValueError(f"Invalid model type: {model_type}")
        model = models[model_type]
        return model.fit_model(x, y, yerr, **kwargs)


class Ephemeris(object):
    """Represents the model ephemeris using transit midpoint data over epochs.

    Parameters
    -----------
    transit_times: TransitTimes obj
        A successfully instantiated TransitTimes object holding epochs, mid transit times, and uncertainties.
        
    Raises
    ----------
     ValueError:
        Raised if transit_times is not an instance of the TransitTimes object.
    """
    def __init__(self, transit_times):
        """Initializing the transit times object and model ephermeris object"""
        self.transit_times = transit_times
        self._validate()

    def _validate(self):
        """Check that transit_times is an instance of the TransitTimes object."""
        if not isinstance(self.transit_times, TransitTimes):
            raise ValueError("Variable 'transit_times' expected type of object 'TransitTimes'.")
        
    def _get_transit_times_data(self):
        """Normalizes transit time data and returns for use."""
        x = self.transit_times.epochs - np.min(self.transit_times.epochs)
        y = self.transit_times.mid_transit_times - np.min(self.transit_times.mid_transit_times)
        yerr = self.transit_times.mid_transit_times_uncertainties
        return x, y, yerr
    
    def _get_model_parameters(self, model_type, **kwargs):
        """Creates the model ephemeris object and returns model parameters.
        
        This method processes and fetches data from the TransitTimes object to be used in the model ephemeris. 
        It creates the appropriate subclass of BaseModelEphemeris using the ModelEphemeris factory, then runs 
        the fit_model method to return the model parameters dictionary.

        Parameters
        ----------
            model_type: str
                Either 'linear' or 'quadratic, the ephemeris subclass specified to create and run.

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
        x, y, yerr = self._get_transit_times_data()
        # Step 2: Create the model with the given variables & user inputs. 
        # This will return a dictionary with the model parameters as key value pairs.
        model_ephemeris_data = ModelEphemerisFactory.create_model(model_type, x, y, yerr, **kwargs)
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
        return np.sqrt((T0_err**2) + ((self.transit_times.epochs**2)*(P_err**2)))
    
    def _calc_quadratic_model_uncertainties(self, T0_err, P_err, dPdE_err):
        """Calculates the uncertainties of a given quadratic model when compared to actual data in TransitTimes.
        
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
        return np.sqrt((T0_err**2) + ((self.transit_times.epochs**2)*(P_err**2)) + ((1/4)*(self.transit_times.epochs**4)*(dPdE_err**2)))
    
    def _calc_linear_ephemeris(self, epochs, period, conjunction_time):
        # NOTE: Is this actually calculating mid transit times?
        """Calculates the mid transit times using parameters from a linear model ephemeris.
        
        Uses the equation (T0 + PE) to calculate the mid transit times over each epoch where T0 is 
        conjunction time, P is period, and E is epoch.

        Parameters
        ----------
            epochs: numpy.ndarray[int]
                The epochs pulled from the TransitTimes object.
            period: float
                The period of the exoplanet transit as calculated by the linear ephemeris model.
            conjunction_time: float
                The conjunction time of the exoplanet transit as calculated by the linear ephemeris model.

        Returns
        -------
            A numpy array of mid transit times calculated over each epoch using the equation above.
        """
        return ((period*epochs) + conjunction_time)
    
    def _calc_quadratic_ephemeris(self, epochs, period, conjunction_time, period_change_by_epoch):
        """Calculates the mid transit times using parameters from a quadratic model ephemeris.
        
        # Returning 

        Uses the equation (T0 + PE + 0.5 * dPdE * E^2) to calculate the mid transit times over each epoch 
        where T0 is conjunction time, P is period, E is epoch, and dPdE is period change with respect to epoch.

        Parameters
        ----------
            epochs: numpy.ndarray[int]
                The epochs pulled from the TransitTimes object.
            period: float
                The period of the exoplanet transit as calculated by the linear ephemeris model.
            conjunction_time: float
                The conjunction time of the exoplanet transit as calculated by the linear ephemeris model.
            period_change_by_epoch: float
                The period change with respect to epoch as calculated by the linear ephemeris model.

        Returns
        -------
            A numpy array of mid transit times calculated over each epoch using the equation above.
        """
        return((0.5*period_change_by_epoch*(epochs**2)) + (period*epochs) + conjunction_time)
    
    def _calc_chi_squared(self, model_data):
        # TODO: Docstring
        """Calculates the residual chi squared values for the 
        
        """
        # TODO: change model_data to something more descriptive
        # STEP 1: Get observed transit times
        observed_data = self.transit_times.mid_transit_times
        uncertainties = self.transit_times.mid_transit_times_uncertainties
        # STEP 2: calculate X2 with observed data and model data
        return np.sum(((observed_data - model_data)/uncertainties)**2)
    
    def get_model_ephemeris(self, model_type):
        """Fits the transit data to a specified model using scipy.optimize.curve_fit function.
        
        Parameters
        ----------
            model_type: str
                Either 'linear' or 'quadratic'. Represents the type of ephemeris to fit the data to.

        Returns
        -------
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
        """
        # NOTE: Are these values returned in days? ex: conjunction time, orbital period, change in orbital period
        parameters = self._get_model_parameters(model_type)
        parameters['model_type'] = model_type
        # Once we get parameters back, we call _calc_linear_ephemeris 
        if model_type == 'linear':
            # Return dict with parameters and model data
            parameters['model_data'] = self._calc_linear_ephemeris(self.transit_times.epochs, parameters['period'], parameters['conjunction_time'])
        elif model_type == 'quadratic':
            parameters['model_data'] = self._calc_quadratic_ephemeris(self.transit_times.epochs, parameters['period'], parameters['conjunction_time'], parameters['period_change_by_epoch'])
        return parameters
    
    def get_ephemeris_uncertainties(self, model_params):
        """Calculates the uncertainties of a specific model data when compared to the actual data. 
        
        Uses the equation σ(t pred, tra) = √(σ(T0)^2 + σ(P)^2 * E^2) for linear models and 
        σ(t pred, tra) = √(σ(T0)^2 + (σ(P)^2 * E^2) + (1/4 * σ(dP/dE)^2 * E^4)) for quadratic models 
        (where σ(T0)=conjunction time error, E=epoch, σ(P)=period error, and σ(dP/dE)=period change by 
        epoch error) to calculate the uncertainties between the model data and actual data over epochs.
        
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
        """Calculates the BIC value for a given model ephemeris. 
        
        Uses the equation BIC = 𝜒2 + (k * log(N)) where 
        𝜒2=∑((observed mid transit times - model mid transit times)/observed mid transit time uncertainties)^2, 
        k=number of fit parameters (2 for linear models, 3 for quadratic models), and N=total number of data points.
        
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
        """Calculates the ΔBIC value between linear and quadratic model ephemerides using the given transit data. 
        
        Returns
        ------- 
            A float value representing the ΔBIC value for this transit data.
        """
        linear_data = self.get_model_ephemeris('linear')
        quadratic_data = self.get_model_ephemeris('quadratic')
        linear_bic = self.calc_bic(linear_data)
        quadratic_bic = self.calc_bic(quadratic_data)
        delta_bic = linear_bic - quadratic_bic
        return delta_bic
    
    def plot_model_ephemeris(self, model_data_dict, save_plot=False, save_filepath=None):
        """Returns a MatplotLib scatter plot showing predicted mid transit times from the model ephemeris over epochs.
        
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
        plt.scatter(x=self.transit_times.epochs, y=model_data_dict['model_data'])
        plt.xlabel('Epochs')
        plt.ylabel('Model Predicted Mid-Transit Times (units)')
        plt.title(f'Predicted {model_data_dict["model_type"]} Model Mid Transit Times over Epochs')
        if save_plot == True:
            plt.savefig(save_filepath)
        plt.show()

    def plot_timing_uncertainties(self, model_data_dict, save_plot=False, save_filepath=None):
        """TODO: This short docstring

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
        x = self.transit_times.epochs
        # get T(E) - T0 - PE
        y = (model_data_dict['model_data'] - model_data_dict['conjunction_time'] - (model_data_dict['period']*self.transit_times.epochs))
        # plot the y line, then the line +- the uncertainties
        plt.plot(x, y, c='blue', label='$t(E) - T_{0} - PE$')
        plt.plot(x, y + model_uncertainties, c='red', label='$(t(E) - T_{0} - PE) + σ_{t^{pred}_{tra}}$')
        plt.plot(x, y - model_uncertainties, c='red', label='$(t(E) - T_{0} - PE) - σ_{t^{pred}_{tra}}$')
        # Add labels and show legend
        plt.xlabel('Epochs')
        plt.ylabel('Days')
        plt.legend()
        if save_plot is True:
            plt.savefig(save_filepath)
        plt.show()

    def plot_oc_plot(self, save_plot=False, save_filepath=None):
        # TODO: Something wrong with this, maybe with data, the y axis values aren't right and the plot for 1/2 dP/dE E^2 looks totally wrong.
        """TODO docstring

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

        quad_model_curve = ((1/2)*quad_model['period_change_by_epoch'])*((self.transit_times.epochs - np.median(self.transit_times.epochs))**2)

        # plot points w/ x=epoch, y=T0-PE, yerr=sigmaT0
        plt.errorbar(self.transit_times.epochs, (self.transit_times.mid_transit_times - lin_model['conjunction_time'] - (lin_model['period']*self.transit_times.epochs)), 
                    yerr=self.transit_times.mid_transit_times_uncertainties, marker='o', ls='', color='#0033A0',
                    label=r'$t(E) - T_0 - P E$')
        plt.plot(self.transit_times.epochs,
                 (quad_model_curve),
                 color='#D64309', label=r'$\frac{1}{2}(\frac{dP}{dE})E^2$')
        plt.legend()
        plt.xlabel('E - Median E')
        plt.ylabel('O-C (seconds)')
        if save_plot is True:
            plt.savefig(save_filepath)
        plt.show()

    def plot_running_delta_bic(self, save_plot=False, save_filepath=None):
        """TODO docstring

        Parameters
        ----------
            save_plot: bool 
                If True, will save the plot as a figure.
            save_filepath: Optional(str)
                The path used to save the plot if `save_plot` is True.
        
        Returns
        -------
            A MatplotLib scatter plot of epochs vs. ΔBIC for each epoch.
        """
        delta_bics = []
        all_epochs = self.transit_times.epochs
        all_mid_transit_times = self.transit_times.mid_transit_times
        all_uncertainties = self.transit_times.mid_transit_times_uncertainties
        # for each epoch (starting at 3?), calculate the delta bic, plot delta bics over epoch
        for i in range(0, len(all_epochs)):
            if i < 2:
                delta_bics.append(int(0))
            else:
                self.transit_times.mid_transit_times = all_mid_transit_times[:i+1]
                self.transit_times.mid_transit_times_uncertainties = all_uncertainties[:i+1]
                self.transit_times.epochs = all_epochs[:i+1]
                delta_bic = self.calc_delta_bic()
                delta_bics.append(delta_bic)
        plt.scatter(x=self.transit_times.epochs, y=delta_bics)
        plt.grid(True)
        plt.plot(self.transit_times.epochs, delta_bics)
        if save_plot is True:
            plt.savefig(save_filepath)
        plt.show()

if __name__ == '__main__':
    # STEP 1: Upload data from file
    filepath = "../../malia_examples/WASP12b_transit_ephemeris.csv"
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    # STEP 2: Break data up into epochs, mid transit times, and error
    # STEP 2.5 (Optional): Make sure the epochs are integers and not floats
    epochs = data["epoch"].astype('int')
    mid_transit_times = data["transit_time"]
    mid_transit_times_err = data["sigma_transit_time"]
    # STEP 3: Create new transit times object with above data
    # transit_times_obj1 = TransitTimes('jd', epochs, mid_transit_times, mid_transit_times_err, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    # print(vars(transit_times_obj1))
    transit_times_obj1 = TransitTimes('jd', epochs, mid_transit_times, mid_transit_times_err, time_scale='tdb')
    print(f"EPOCHS: {transit_times_obj1.epochs}\n")
    print(f"MID TRANSIT TIMES: {transit_times_obj1.mid_transit_times}\n")
    # STEP 4: Create new ephemeris object with transit times object
    ephemeris_obj1 = Ephemeris(transit_times_obj1)
    # STEP 5: Get model ephemeris data
    linear_model_data = ephemeris_obj1.get_model_ephemeris('linear')
    quad_model_data = ephemeris_obj1.get_model_ephemeris('quadratic')
    print(linear_model_data)
    print(quad_model_data)
    # ephemeris_obj1.plot_model_ephemeris(linear_model_data)
    # ephemeris_obj1.plot_model_ephemeris(quad_model_data)
    ephemeris_obj1.plot_oc_plot()
    # model_uncertainties = ephemeris_obj1.get_ephemeris_uncertainties(model_data)
    # print(model_uncertainties)
    # # STEP 6: Show a plot of the model ephemeris data
    # # ephemeris_obj1.plot_model_ephemeris(model_data, save_plot=False)
    # # # STEP 7: Uncertainties plot
    # # ephemeris_obj1.plot_timing_uncertainties(model_data, save_plot=False)
    # # bic = ephemeris_obj1.calc_bic(model_data)
    # # print(bic)

    # # print(ephemeris_obj1.calc_delta_bic())
    # print(ephemeris_obj1.plot_running_delta_bic(save_plot=False))
    # # ephemeris_obj1.plot_running_delta_bic(save_plot=False)

    # # ephemeris_obj1.plot_oc_plot(False)
