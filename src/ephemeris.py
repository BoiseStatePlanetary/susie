from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
import numpy as np
from transit_times import TransitTimes

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
            'period_change_by_epoch_err': unc[2]
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
        # Check that if model_ephemerides is given, it is a list of ModelEphemeris objects **will do if we keep it that way
        
    def get_model_parameters(self, model_type, **kwargs):
        # NOTE: Do we want to return the model ephemeris object to the user or just the return data dict? *Would help to keep in mind how users may use this going further into package use
        # Step 1: Get data from transit times obj
        x = self.transit_times.epochs - np.min(self.transit_times.epochs)
        y = self.transit_times.mid_transit_times - np.min(self.transit_times.mid_transit_times)
        yerr = self.transit_times.uncertainties
        # Step 2: Create the model with the given variables & user inputs
        model_ephemeris = ModelEphemerisFactory.create_model(model_type, x, y, yerr, **kwargs) # this will return the model parameters as dict
        # Step 3: Iterate over every key value pair in the return data dictionary and add as attribute to this object
        # The model_ephemeris variable will return a dictionary of model parameters and their errors
        # an example of this would look like:
        # {'period': 329847, 'period_err': 2931, 'conjunction': 123231, 'conjunction_err': 2183}
        for key, val in model_ephemeris.items():
            setattr(self, key, val)
        # Step 4: Return the return data dictionary to the user just so they can see what's going on
        return model_ephemeris

    def get_bic(self):
        # TODO: Figure out how to calculate this and what we need from the user ***FROM UTILS IN BRIAN CODE
        # Step 1: Get value of k based on model_type (linear=2, quad=3, custom=?)
        # Step 2: Calculate chi-squared
        # Step 3: Calculate BIC
        pass

# if __name__ == '__main__':
#     data = np.genfromtxt("./malia_examples/WASP12b_transit_ephemeris.csv", delimiter=',', names=True)
#     epochs = data["epoch"] - np.min(data["epoch"])
#     mid_transit_times = data["transit_time"] - np.min(data["transit_time"])
#     uncertainties = data["sigma_transit_time"]
#     tt1 = TransitTimes(epochs, mid_transit_times, uncertainties)
