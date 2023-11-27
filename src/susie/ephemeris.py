from abc import ABC, abstractmethod
from scipy.optimize import curve_fit
import numpy as np
import math
from src.susie.transit_times import TransitTimes

class BaseModelEphemeris(ABC):
    '''
    Creates a holding place for the fit variables (x, y, yerr, \*\*kwargs) before a linear, quadratic, or custom fit can be created. 
    '''
    @abstractmethod
    def fit_model(self, x, y, yerr, **kwargs):
        pass


class LinearModelEphemeris(BaseModelEphemeris):
    '''
    Generates a linear fit to the model ephemeris data. 

    Parameters
    ----------
        BaseModelEphemeris : class
            a holding place for x, y, and yerr

    STEP 1: create a linear fit model by calling lin_fit.
    
    STEP 2: fit the model ephemieris data to the linear fit using SciPy curve fit techniques by calling fit_model.

    Returns
    ------- 
        Return data : dictionary
            contains the period and conjunction time values and errors, which is used to build a best fit linear regression for the model ephemeris data.
    '''
    def lin_fit(self, x, P, T0):
        '''
        creates an initial linear fit model based off of the first data point in ??? - I think I need to review the paper - do this week of 10/30

        Parameters
        ----------
            x : ??
                The mid-transit time
            p : ??
                The period of a transit ?? 
            T0 : ??
                the initial epoch associated with a mid-transit time
        Returns
        -------
            P*x + T0 :
                a linear function based off of the inital data points in ???
        '''
        return P*x + T0
    
    def fit_model(self, x, y, yerr, **kwargs):
        '''
        Compares the model ephemieris data to the linear fit created by the inital data and creates a curve fit which minimizes the difference between the two data sets. 

        Parameters
        ----------
            x : ??
                The (list?) of epochs, possibly normalized to start at 0 (Question for Malia: has this been normalized to 0 yet?)
            y : ??
                The mid-transit times of a given epoch
            yerr : ??
                The error in the mid-transit time  
            **kwargs : ??
                define this

        Returns
        ------- 
        Return data : dictionary
            contains the period and conjunction time values and errors, which is used to build a best fit linear regression for the model ephemeris data. 
        '''
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
    '''
    Generates a quadratic fit to the model ephemeris data. 

    Parameters
    ---------- 
        BaseModelEphemeris : class
            a holding place for x, y, and yerr

    STEP 1: create a quadratic fit model.
    
    STEP 2: fit the model ephemieris data to the quadratic fit using SciPy curve fit techniques.

    Returns
    ------- 
        Return data : dictionary
            contains the period, conjunction time, and the period change by epoch values and errors, which is used to build a best fit quadratic regression for the model ephemeris data.
    '''
    def quad_fit(self, x, dPdE, P, T0):
        '''
        creates an initial quadratic fit model based off of the first data point in ??? - I think I need to review the paper - do this week of 10/30

        Parameters
        ----------
            x : ??
                The mid-transit time
            dPdE : ??
                change in period (maybe) with respect to ???
            p : ??
                The period of the transit time, in days??
            T0 : ??
                The initial epoch associated with a mid-transit time???
        Returns
        -------
            0.5*dPdE*x*x + P*x + T0 : ??
                a quadratic function based off of the inital data points in ???
        '''
        return 0.5*dPdE*x*x + P*x + T0
    
    def fit_model(self, x, y, yerr, **kwargs):
        '''
        Compares the model ephemieris data to the  fit created by the inital data and creates a curve fit which minimizes the difference between the two data sets. 

        Parameters
        ----------
            x : ??
                The (list?) of epochs, possibly normalized to start at 0 (Question for Malia: has this been normalized to 0 yet?)
            y : ??
                The mid-transit times of a given epoch
            yerr : ??
                The error in the mid-transit time
            **kwargs : ??
                define this

        Returns
        ------- 
        Return data : dictionary
            contains the period, conjunction time, and period change by epoch values and errors, which is used to build a best fit quadratic regression for the model ephemeris data. 
        '''
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
    '''
    I think this will get changed to accept more user input - talk to malia at a later date - HV 10/22/23
    '''
    def fit_model(self, x, y, yerr, **kwargs):
        pass


class ModelEphemerisFactory:
    '''
    Selects which type of SciPy model (linear, quadratic, or custom) will be used to model the given ephmeris data

    Parameters
    ----------
        SHOULD WE STATE ANY : 


    STEP 1: depending on the model type, call LinearModelEphemeris(), QuadraticModelEphemeris(), or CustomModelEphemeris(). 

    Returns
    ------- 
        model : dictonary
            the returned data dictionary from whichever regression type (linear, quadratic, or custom) was called.

    Raises
    ------ 
        error if model specified is not 'linear', 'quadratic', or 'custom'.
    '''
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
    """BIG DEFINITION 

    Parameters
    -----------
    transit_times : obj
        user generated object from transit_times.py
    model_type : str 
        Specifies linear or quadratic model of transit time

    Returns
    -------
    model_ephemeris : list 
        Model of epemieris as a list of dictionaries
    
    Raises
    ------
        ValueError raised if transit_times is not an instance of the TransitTimes object
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
        docstring here

        Parameters
        ----------
            model_type : string
                Type of ephemeris model (either 'linear' or 'quadratic') to build

        STEP 1: Call get model parameters to create a model ephemeris object. get model parameters returns a data dictionary containing data from transit times (epochs, mid transit times, and error). If an error is not given, we will substitute our own error using an array of 1s in the same shape as the given mid transit itmes.
            
        STEP 2: Call calc linear ephemeris or calc quadratic ephemeris depending on the model type. Both models use SciPy curve fit to fit data to either a linear or quadratic curve and then returns the curve. 
                     
        Returns
        ------- 
            model_data : dictionary
                model parameters needed to build the curve fit for the data.
        """
        # Try to get parameters (parameters if linear=period, conjunction time, if quadratic=period, conjunction time, period change by epoch)
        # EXPLANATION:
            # call _get_model... pass in model type (linear or quadratic) model
            # Getting data from transit times (epochs, mid transit times, and error) ** If an error is not given, we will substitute our own error using an array of 1s in the same shape as your given mid transit itmes
            # Creates the model ephemeris with ModelEPhemerisFactory obj, passing in data
            # Model factory will choose which model object to create based on model_type
            # Model will use scipy curve fit to fit data to whatever and then return parameters
            # RETURNS model parameters as a dictionary
        parameters = self._get_model_parameters(model_type)
        # ONce we get parameters back, we call _calc_linear_ephemeris 
        if model_type == 'linear':
            return self._calc_linear_ephemeris(self.transit_times.epochs, parameters['period'], parameters['conjunction_time'])
        elif model_type == 'quadratic':
            return self._calc_quadratic_ephemeris(self.transit_times.epochs, parameters['period'], parameters['conjunction_time'], parameters['period_change_by_epoch'])
    
    def calc_chi_squared(self):
        """
        needs to be built
        """
        # STEP 1: Calculate model data using given epochs (these will be predicted transit times)
        # STEP 2: Get observed transit times
        # STEP 3: calculate X2
        # NOTE: Do we want this to be connected to a 
        # return np.sum(((given_data - model_data)/uncertainties)**2)
        pass

    def calc_bic(self):
        """
         needs to be built
        """
        # Step 1: Get value of k based on model_type (linear=2, quad=3, custom=?)
        # Step 2: Calculate chi-squared
        # Step 3: Calculate BIC
        # chi_sq = calc_chi_sq(data, model, sigma)
        # return chi_sq + num_params*np.log(len(data))
        pass

    def calc_delta_bic(self):
        """
         needs to be built
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

