from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt
from susie.transit_times import TransitTimes
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
import requests

class BaseModelEphemeris(ABC):
    @abstractmethod
    def fit_model(self, x, y, yerr, **kwargs):
        pass


class LinearModelEphemeris(BaseModelEphemeris):
    def lin_fit(self, x, P, T0):
        return P*x + T0
    
    def fit_model(self, x, y, yerr, **kwargs):
        popt, pcov = curve_fit(self.lin_fit, x, y, sigma=yerr, absolute_sigma=True, **kwargs)
        unc = np.sqrt(np.diag(pcov))
        print(f'uncertainty: {unc}')
        return_data = {
            'period': popt[0],
            'period_err': unc[0],
            'conjunction_time': popt[1],
            'conjunction_time_err': unc[1]
        }
        return(return_data)


class QuadraticModelEphemeris(BaseModelEphemeris):
    def quad_fit(self, x, dPdE, P, T0):
        return 0.5*dPdE*x*x + P*x + T0
    
    def fit_model(self, x, y, yerr, **kwargs):
        popt, pcov = curve_fit(self.quad_fit, x, y, sigma=yerr, absolute_sigma=True, **kwargs)
        unc = np.sqrt(np.diag(pcov))
        return_data = {
            'period': popt[0],
            'period_err': unc[0],
            'conjunction_time': popt[1],
            'conjunction_time_err': unc[1],
            'period_change_by_epoch': popt[2],
            'period_change_by_epoch_err': unc[2],
        }
        return(return_data)


class CustomModelEphemeris(BaseModelEphemeris):
    def fit_model(self, x, y, yerr, **kwargs):
        pass


class ModelEphemerisFactory:
    @staticmethod
    def create_model(model_type, x, y, yerr, **kwargs):
        models = {
            'linear': LinearModelEphemeris(),
            'quadratic': QuadraticModelEphemeris(),
            'custom': CustomModelEphemeris()
        }

        if model_type not in models:
            raise ValueError(f"Invalid model type: {model_type}")

        model = models[model_type]
        return model.fit_model(x, y, yerr, **kwargs)


class Ephemeris(object):
    """Docstring about the ephemeris object.

    Parameters
    -----------
    transit_times : obj
        user generated object from transit_times.py
    model_type : str 
        Specifies linear or quadratic model of transit time

        
    Returns
    -------
    model_ephemeris : list - ask
        Model of epemieris as a list of dictionaries
    
    Exceptions
    ----------
     ValueError
        raised if transit_times is not an instance of the TransitTimes object
    ----------

    """
    def __init__(self, transit_times):
        # initializing the transit times object and model ephermeris object
        # NOTE: We will have the user create their own transit times obj and send that to the ephemeris obj
        self.transit_times = transit_times
        self._validate()

    def _validate(self):
        # Check that transit_times is an instance of the TransitTimes object
        if not isinstance(self.transit_times, TransitTimes):
            raise ValueError("Variable 'transit_times' expected type of object 'TransitTimes'.")
        
    def _get_transit_times_data(self):
        # Process the transit data so it can be used
        # NOTE QUESTION: Why do we subtract np.min?
        x = self.transit_times.epochs - np.min(self.transit_times.epochs)
        y = self.transit_times.mid_transit_times - np.min(self.transit_times.mid_transit_times)
        yerr = self.transit_times.mid_transit_times_uncertainties
        return x, y, yerr
    
    def _get_model_parameters(self, model_type, **kwargs):
        # Step 1: Get data from transit times obj
        x, y, yerr = self._get_transit_times_data()
        # Step 2: Create the model with the given variables & user inputs. 
        # This will return a dictionary with the model parameters as key value pairs.
        model_ephemeris_data = ModelEphemerisFactory.create_model(model_type, x, y, yerr, **kwargs)
        # Step 3: Return the data dictionary with the model parameters
        return model_ephemeris_data
    
    def _get_k_value(self, model_type):
        if model_type == 'linear':
            return 2
        elif model_type == 'quadratic':
            return 3
        else:
            return ValueError('Only linear and quadratic models are supported at this time.')
    
    def _calc_linear_model_uncertainties(self, model_data):
        # σ(t pred, tra) = √σ(T0)^2 + σ(P)^2 * E^2
        # if I want to return in seconds, multiply conjunction time and period by: days > 24 hours > 60 min > 60 sec ()
        # c_time = model_data['conjunction_time_err'] * 24 * 60 * 60
        # period = model_data['period_err'] * 24 * 60 * 60
        # print('new c time', c_time, 'new period', period)
        # return np.sqrt((c_time**2) + ((self.transit_times.epochs**2)*(period**2)))
        return np.sqrt((model_data['conjunction_time_err']**2) + ((self.transit_times.epochs**2)*(model_data['period_err']**2)))
    
    def _calc_quadratic_model_uncertainties(self, model_data):
        # σ(t pred, tra) = √σ(T0)^2 + (σ(P)^2 * E^2) + (1/4 * σ(dP/dE)^2 * E^4)
        # if I want to return in second, 
        return np.sqrt((model_data['conjunction_time_err']**2) + ((self.transit_times.epochs**2)*(model_data['period_err']**2)) + ((1/4)*(self.transit_times.epochs**4)*(model_data['period_change_by_epoch_err']**2)))
    
    def _calc_linear_ephemeris(self, epochs, period, conjunction_time):
        return ((period*epochs) + conjunction_time)
    
    def _calc_quadratic_ephemeris(self, epochs, period, conjunction_time, period_change_by_epoch):
        return((0.5*period_change_by_epoch*(epochs**2)) + (period*epochs) + conjunction_time)
    
    def _calc_chi_squared(self, model_data):
        """

        """
        # STEP 1: Get observed transit times
        observed_data = self.transit_times.mid_transit_times
        uncertainties = self.transit_times.mid_transit_times_uncertainties
        # STEP 2: calculate X2 with observed data and model data
        return np.sum(((observed_data - model_data)/uncertainties)**2)
    
    def get_model_ephemeris(self, model_type):
        # Returns predicted transit times for given epochs
        # EXPLANATION:
            # call _get_model... pass in model type (linear or quadratic) 
            # Getting data from transit times (epochs, mid transit times, and error) ** If an error is not given, we will substitute our own error using an array of 1s in the same shape as your given mid transit itmes
            # Creates the model ephemeris with ModelEPhemerisFactory obj, passing in data
            # Model factory will choose which model object to create based on model_type
            # Model will use scipy curve fit to fit data to whatever and then return parameters
            # RETURNS model parameters as a dictionary
            # NOTE: Are these values returned in days? ex: conjunction time, orbital period, change in orbital period
        parameters = self._get_model_parameters(model_type)
        parameters['model_type'] = model_type
        # ONce we get parameters back, we call _cal_linear_ephemeris 
        if model_type == 'linear':
            # Return dict with parameters and model data
            parameters['model_data'] = self._calc_linear_ephemeris(self.transit_times.epochs, parameters['period'], parameters['conjunction_time'])
        elif model_type == 'quadratic':
            parameters['model_data'] = self._calc_quadratic_ephemeris(self.transit_times.epochs, parameters['period'], parameters['conjunction_time'], parameters['period_change_by_epoch'])
        return parameters
    
    def get_ephemeris_uncertainties(self, model_data):
        if model_data['model_type'] == 'linear':
            return self._calc_linear_model_uncertainties(model_data)
        elif model_data['model_type'] == 'quadratic':
            return self._calc_quadratic_model_uncertainties(model_data)
    
    def calc_bic(self, model_data_dict):
        """
        """
        # Step 1: Get value of k based on model_type (linear=2, quad=3, custom=?)
        num_params = self._get_k_value(model_data_dict['model_type'])
        # Step 2: Calculate chi-squared
        chi_squared = self._calc_chi_squared(model_data_dict['model_data'])
        # Step 3: Calculate BIC
        return chi_squared + (num_params*np.log(len(model_data_dict['model_data'])))

    def calc_delta_bic(self):
        """
        """
        linear_data = self.get_model_ephemeris('linear')
        quadratic_data = self.get_model_ephemeris('quadratic')
        linear_bic = self.calc_bic(linear_data)
        quadratic_bic = self.calc_bic(quadratic_data)
        delta_bic = linear_bic - quadratic_bic
        return delta_bic
    
    def plot_model_ephemeris(self, model_data_dict, save_plot):
        plt.scatter(x=self.transit_times.epochs, y=model_data_dict['model_data'])
        plt.xlabel('Epochs')
        plt.ylabel('Model Predicted Mid-Transit Times')
        plt.title('Predicted Model Mid Transit Times over Epochs')
        if save_plot == True:
            plt.savefig()
        plt.show()

    def plot_timing_uncertainties(self, model_data, save_plot):
        # get uncertainties
        model_uncertainties = self.get_ephemeris_uncertainties(model_data)
        x = self.transit_times.epochs
        # get T(E) - T0 - PE
        y = (model_data['model_data'] - model_data['conjunction_time'] - (model_data['period']*self.transit_times.epochs))
        # plot the y line, then the line +- the uncertainties
        plt.plot(x, y, c='blue', label='$t(E) - T_{0} - PE$')
        plt.plot(x, y + model_uncertainties, c='red', label='$(t(E) - T_{0} - PE) + σ_{t^{pred}_{tra}}$')
        plt.plot(x, y - model_uncertainties, c='red', label='$(t(E) - T_{0} - PE) - σ_{t^{pred}_{tra}}$')
        # Add labels and show legend
        plt.xlabel('Epochs')
        plt.ylabel('Days')
        plt.legend()
        if save_plot is True:
            plt.savefig()
        plt.show()

    def plot_oc_plot(self, model_data, save_plot):
        pass

    def plot_running_delta_bic(self, save_plot):
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
        plt.plot(self.transit_times.epochs, delta_bics)
        plt.show()



if __name__ == '__main__':
    # STEP 1: Upload data from file
    filepath = "../../malia_examples/WASP12b_transit_ephemeris.csv"
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    # STEP 2: Break data up into epochs, mid transit times, and error
    epochs = data["epoch"] - np.min(data["epoch"])
    mid_transit_times = data["transit_time"] - np.min(data["transit_time"])
    mid_transit_times_err = data["sigma_transit_time"]
    # NOTE: You can use any method and/or file type to upload your data. Just make sure 
    # the resulting variables (epoch, mid transit times, and mid transit time errors) 
    # are numpy arrays
    # STEP 2.5 (Optional): Make sure the epochs are integers and not floats
    epochs = epochs.astype('int')
    # STEP 3: Create new transit times object with above data
    transit_times_obj1 = TransitTimes(epochs, mid_transit_times, mid_transit_times_err)
    # STEP 4: Create new ephemeris object with transit times object
    ephemeris_obj1 = Ephemeris(transit_times_obj1)
    # STEP 5: Get model ephemeris data
    model_data = ephemeris_obj1.get_model_ephemeris('quadratic')
    print(model_data)
    # STEP 6: Show a plot of the model ephemeris data
    # ephemeris_obj1.plot_model_ephemeris(model_data, save_plot=False)
    # STEP 7: Uncertainties plot
    # ephemeris_obj1.plot_timing_uncertainties(model_data, save_plot=False)
    bic = ephemeris_obj1.calc_bic(model_data)
    print(bic)

    print(ephemeris_obj1.calc_delta_bic())
    print(ephemeris_obj1.plot_running_delta_bic(save_plot=False))
