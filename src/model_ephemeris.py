from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import curve_fit

class BaseModelEphemeris(ABC):
    @abstractmethod
    def fit_model(self, x, y, yerr, **kwargs):
        pass

    def lin_fit(self, x, P, T0):
        return P*x + T0

    def quad_fit(self, x, dPdE, P, T0):
        return 0.5*dPdE*x*x + P*x + T0


class LinearModelEphemeris(BaseModelEphemeris):
    def fit_model(self, x, y, yerr, **kwargs):
        popt, pcov = curve_fit(self.lin_fit, x, y, sigma=yerr, absolute_sigma=True, **kwargs)
        unc = np.sqrt(np.diag(pcov))
        print("curve_fit linear")
        print("P = (%g +- %g) days" % (popt[0], unc[0]))
        print("T0 = (%g +- %g) days" % (popt[1], unc[1]))
        return_data = {
            'period': popt[0],
            'period_err': unc[0],
            'conjunction_time': popt[1],
            'conjunction_time_err': unc[1]
        }
        return(return_data)


class QuadraticModelEphemeris(BaseModelEphemeris):
    def fit_model(self, x, y, yerr, **kwargs):
        popt, pcov = curve_fit(self.quad_fit, x, y, sigma=yerr, absolute_sigma=True, **kwargs)
        unc = np.sqrt(np.diag(pcov))
        print("curve_fit quadratic")
        print("dPdE = (%g +- %g) days" % (popt[0], unc[0]))
        print("P = (%g +- %g) days" % (popt[1], unc[1]))
        print("T0 = (%g +- %g) days" % (popt[2], unc[2]))
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


class ModelEphemeris:
    def __init__(self):
        pass

    def get_model_parameters(self, model_type, x, y, yerr, **kwargs):
        # Step 1: Instantiate model factory object
        # Step 2: Create the model with the given variables & user inputs
        # Step 3: The model_ephemeris variable will return a dictionary of model parameters and their errors
        # an example of this would look like:
        # {'period': 329847, 'period_err': 2931, 'conjunction': 123231, 'conjunction_err': 2183}
        # Iterate over every key value pair in the return data dictionary and add as attribute to this object
        # Step 4: Return the return data dictionary to the user just so they can see what's going on
        factory = ModelEphemerisFactory()
        model_ephemeris = factory.create_model(model_type, x, y, yerr, **kwargs)
        for key, val in model_ephemeris.items():
            setattr(self, key, val)
        return model_ephemeris

    def get_bic(self):
        # TODO: Figure out how to calculate this and what we need from the user
        # Step 1: Get value of k based on model_type (linear=2, quad=3, custom=?)
        # Step 2: Calculate chi-squared
        # Step 3: Calculate BIC
        pass
