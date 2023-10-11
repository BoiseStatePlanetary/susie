from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
import numpy as np
import math
from susie.transit_times import TransitTimes

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
    """example using this: 

    factory = ModelEphemerisFactory()
    linear_model = factory.create_model('linear', x=epochs, y=transit_time, yerr=uncertainties, fit_intercept=True)
    quadratic_model = factory.create_model('quadratic', x=epochs, y=transit_time, yerr=uncertainties)
    """
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
    
    def _calc_linear_ephemeris(self, epochs, period, conjunction_time):
        return ((period*epochs) + conjunction_time)
    
    def _calc_quadratic_ephemeris(self, epochs, period, conjunction_time, period_change_by_epoch):
        return ((pow((period_change_by_epoch*epochs), 2)) + (period*epochs) + conjunction_time)
    
    def get_model_ephemeris(self, model_type):
        # Returns predicted transit times for given epochs
        """
            STEP 1: Call get model parameters to create a model ephemeris object

            Parameters:
                model_type: string
                    Type of ephemeris model (either 'linear' or 'quadratic') to build

            Returns:

            

        """
        parameters = self._get_model_parameters(model_type)
        if model_type == 'linear':

            return self._calc_linear_ephemeris(self.transit_times.epochs, parameters['period'], parameters['conjunction_time'])
        elif model_type == 'quadratic':
            return self._calc_quadratic_ephemeris(self.transit_times.epochs, parameters['period'], parameters['conjunction_time'], parameters['period_change_by_epoch'])
    
    def calc_chi_squared(self):
        """
        
        """
        # STEP 1: Calculate model data using given epochs (these will be predicted transit times)
        # STEP 2: Get observed transit times
        # STEP 3: calculate X2
        # NOTE: Do we want this to be connected to a 
        # return np.sum(((given_data - model_data)/uncertainties)**2)
        pass

    def calc_bic(self):
        """
        """
        # Step 1: Get value of k based on model_type (linear=2, quad=3, custom=?)
        # Step 2: Calculate chi-squared
        # Step 3: Calculate BIC
        # chi_sq = calc_chi_sq(data, model, sigma)
        # return chi_sq + num_params*np.log(len(data))
        pass

    def calc_delta_bic(self):
        """
        """
        pass



if __name__ == '__main__':
    # STEP 1: Upload data from file
    filepath = "./malia_examples/WASP12b_transit_ephemeris.csv"
    data = np.genfromtxt(filepath, delimiter=',', names=True)
    # STEP 2: Break data up into epochs, mid transit times, and error
    epochs = data["epoch"] - np.min(data["epoch"])
    mid_transit_times = data["transit_time"] - np.min(data["transit_time"])
    mid_transit_times_err = data["sigma_transit_time"]
    # NOTE: You can use any method and/or file type to upload your data. Just make sure the resulting variables (epoch, mid transit times, and mid transit time errors) are numpy arrays
    # STEP 2.5 (Optional): Make sure the epochs are integers and not floats
    epochs = epochs.astype('int')
    # STEP 3: Create new transit times object with above data
    transit_times_obj1 = TransitTimes(epochs, mid_transit_times, mid_transit_times_err)
    # print(vars(transit_times_obj1))
    ephemeris_obj1 = Ephemeris(transit_times_obj1)
    model_data = ephemeris_obj1.get_model_ephemeris('linear')
    print(model_data)
    type(transit_times_obj1)

