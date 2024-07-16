from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from lmfit import Model
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astroplan import FixedTarget, Observer, EclipsingSystem
# from susie.timing_data import TimingData # Use this for package pushes
from timing_data import TimingData # Use this for running tests
# from timing_data import TimingData # Use this for running this file

class Ephemeris():
    """Represents the model ephemeris using transit or occultation mid-time data over epochs.

    Parameters
    -----------
    timing_data: :class:`TimingData`
        A successfully instantiated :class:`TimingData` object holding epochs, mid-times, and uncertainties.
        
    Raises
    ----------
    ValueError:
        Raised if ``timing_data`` is not an instance of the :class:`TimingData` object.
    """
    def __init__(self, timing_data):
        """Initializing the model ephemeris object.

        Parameters
        -----------
        timing_data: :class:`TimingData`
            A successfully instantiated :class:`TimingData` object holding epochs, mid-times, and uncertainties.
        
        Raises
        ------
        ValueError :
            error raised if ``timing_data`` is not an instance of :class:`TimingData` object.
        """
        self.timing_data = timing_data
        self._validate()
    
    def get_model_ephemeris(self, model_type):
        """Fits the timing data to a specified model using an LMfit Model fit.

        Parameters
        ----------
        model_type: str
            Either 'linear' or 'quadratic'. Represents the type of ephemeris to fit the data to.

        Returns
        -------
        model_ephemeris_data: dict
            A dictionary of parameters returned by the fit model ephemeris.
            
            If a linear model was chosen, these parameters are:

            - ``period``: Estimated orbital period of the exoplanet (in units of days)
            - ``period_err``: Uncertainty associated with ``period`` (in units of days)
            - ``conjunction_time``: Time of conjunction of exoplanet transit or occultation (in Julian Date format)
            - ``conjunction_time_err``: Uncertainty associated with ``conjunction_time``

            If a quadratic model was chosen, the same variables are returned, and two additional parameters are included in the dictionary:

            - ``period_change_by_epoch``: The exoplanet period change with respect to epoch (in units of days)
            - ``period_change_by_epoch_err``: The uncertainties associated with ``period_change_by_epoch`` (in units of days)

            Example:
                Here is an example of the dictionary returned for a quadratic model using WASP-12 b data.

                .. code-block:: python

                    {
                        "period": 1.0914217235792794,
                        "period_err": 1.5396843425628388e-07,
                        "conjunction_time": 2454515.525511919,
                        "conjunction_time_err": 0.00014522035702285087,
                        "period_change_by_epoch": -9.89824903833347e-10,
                        "period_change_by_epoch_err": 7.180878117743563e-11
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

        Calculate the uncertainties between the model data and actual data over epochs using the following 
        equations:
        
        For linear models:
        
        - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * E^2)}` for transit observations
        - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * (\\frac{1}{2} + E)^2)}` for occultation observations
            
        For quadratic models:

        - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * E^2) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))}` for transit observations
        - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * (\\frac{1}{2} + E^2)) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))}` for occultation observations
        
        where :math:`\\sigma(T_0) =` conjunction time error, :math:`E=` epoch, :math:`\\sigma(P)=` period error, and :math:`\\sigma(\\frac{dP}{dE})=` period change with respect to epoch error.
        
        Parameters
        ----------
        model_params: dict
            The model ephemeris data dictionary recieved from :meth:`Ephemeris.get_model_ephemeris`.
        
        Returns
        -------
        A list of mid-time uncertainties associated with the model ephemeris passed in, calculated with the 
        equation(s) above and the :class:`TimingData` epochs.
        
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
        Uses the equation:
        

        :math:`BIC = \\chi^2 + (k * log(N))` 
        

        where :math:`\\chi^2=\\sum{\\frac{(\\text{observed midtimes - model midtimes})}{\\text{(observed 
        midtime uncertainties})^2}}`, :math:`k` is the number of fit parameters (2 for linear models, 3 for 
        quadratic models), and :math:`N` is the total number of data points.
        
        Parameters
        ----------
        model_data_dict: dict
            The model ephemeris data dictionary recieved from :meth:`Ephemeris.get_model_ephemeris`.
        
        Returns
        ------- 
        bic: float
            The BIC value for this model ephemeris.
        """
        # Step 1: Get value of k based on model_type (linear=2, quad=3, custom=?)
        num_params = self._get_k_value(model_data_dict['model_type'])
        # Step 2: Calculate chi-squared
        chi_squared = self._calc_chi_squared(model_data_dict['model_data'])
        # Step 3: Calculate BIC
        bic = chi_squared + (num_params*np.log(len(model_data_dict['model_data'])))
        return bic

    def calc_delta_bic(self):
        """Calculates the :math:`\\Delta BIC` value between linear and quadratic model ephemerides using the given timing data. 
        
        The BIC value is a modified :math:`\\chi^2` value that penalizes for additional parameters. The
        :math:`\\Delta BIC` value is the difference between the linear BIC value and the quadratic BIC value.
        Models that have smaller values of BIC are favored. Therefore, if the :math:`\\Delta BIC` value for your
        data is a large positive number (large linear BIC - small quadratic BIC), the quadratic model is favored and
        your data indicates possible orbital decay in your extrasolar system. If the :math:`\\Delta BIC` value for
        your data is a small number or negative (small linear BIC - large quadratic BIC), then the linear model is
        favored and your data may not indicate orbital decay. 

        This function will run all model ephemeris instantiation and BIC calculations for you using the :class:`TimingData`
        information you entered.

        Returns
        ------- 
        delta_bic: float
            Represents the :math:`\\Delta BIC` value for this timing data.
        """
        linear_data = self.get_model_ephemeris('linear')
        quadratic_data = self.get_model_ephemeris('quadratic')
        linear_bic = self.calc_bic(linear_data)
        quadratic_bic = self.calc_bic(quadratic_data)
        delta_bic = linear_bic - quadratic_bic
        return delta_bic
    
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
    
    def plot_model_ephemeris(self, model_data_dict, save_plot=False, fname=None):
        """Plots a scatterplot of epochs vs. model calculated mid-times.

        Parameters
        ----------
            model_data_dict: dict
                The model ephemeris data dictionary recieved from the `get_model_ephemeris` method.
            save_plot: bool 
                If True, will save the plot as a figure.
            fname: Optional(str)
                The path used to save the plot if `save_plot` is True.
        """
        plt.scatter(x=self.timing_data.epochs, y=model_data_dict['model_data'], color='#0033A0')
        plt.xlabel('Epochs')
        plt.ylabel('Model Predicted Mid-Times (units)')
        plt.title(f'Predicted {model_data_dict["model_type"].capitalize()} Model Mid Times over Epochs')
        if save_plot == True:
            plt.savefig(fname)
        plt.show()

    def plot_timing_uncertainties(self, model_data_dict, save_plot=False, fname=None):
        """Plots a scatterplot of epochs vs. model calculated mid-time uncertainties.

        Parameters
        ----------
            model_data_dict: dict
                The model ephemeris data dictionary recieved from the `get_model_ephemeris` method.
            save_plot: bool 
                If True, will save the plot as a figure.
            fname: Optional(str)
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
            plt.savefig(fname)
        plt.show()

    def plot_oc_plot(self, save_plot=False, fname=None):
        """Plots a scatter plot of observed minus calculated values of mid-times for linear and quadratic model ephemerides over epochs.

        Parameters
        ----------
            save_plot: bool 
                If True, will save the plot as a figure.
            fname: Optional(str)
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
            plt.savefig(fname)
        plt.show()

    def plot_running_delta_bic(self, save_plot=False, fname=None):
        """Plots a scatterplot of epochs vs. :math:`\\Delta BIC` for each epoch.

        Starting at the third epoch, will plot the value of :math:`\\Delta BIC` for all previous epochs,
        showing how the value of :math:`\\Delta BIC` progresses over time with more observations.

        Parameters
        ----------
            save_plot: bool 
                If True, will save the plot as a figure.
            fname: Optional(str)
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
            plt.savefig(fname)
        plt.show()
        
    def _get_timing_data(self):
        """Returns timing data for use.

        Returns the epoch, mid-time, and mid-time uncertainty data from the :class:`TimingData` object.

        Returns
        -------
        x: numpy.ndarray[int]
            The epoch data as recieved from the :class:`TimingData` object.
        y: numpy.ndarray[float]
            The mid-time data as recieved from the :class:`TimingData` object.
        yerr: numpy.ndarray[float]
            The mid-time error data as recieved from the :class:`TimingData` object.
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
        
        This method fetches data from the :class:`TimingData` object to be used in the model ephemeris. 
        It creates the appropriate subclass of :class:`BaseModelEphemeris` using the :class:`ModelEphemerisFactory`, 
        then runs the ``fit_model`` method to return the model parameters dictionary to the user.

        Parameters
        ----------
        model_type: str
            Either 'linear' or 'quadratic'. The ephemeris subclass specified to create and run.

        Returns
        -------
        model_ephemeris_data: dict
            A dictionary of parameters returned by the fit model ephemeris.
            
            If a linear model was chosen, these parameters are:

            - ``period``: Estimated orbital period of the exoplanet (in units of days)
            - ``period_err``: Uncertainty associated with ``period`` (in units of days)
            - ``conjunction_time``: Time of conjunction of exoplanet transit or occultation (in Julian Date format)
            - ``conjunction_time_err``: Uncertainty associated with ``conjunction_time``

            If a quadratic model was chosen, the same variables are returned, and two additional parameters are included in the dictionary:

            - ``period_change_by_epoch``: The exoplanet period change with respect to epoch (in units of days)
            - ``period_change_by_epoch_err``: The uncertainties associated with ``period_change_by_epoch`` (in units of days)

            Example:
                Here is an example of the dictionary returned for a quadratic model using WASP-12 b data.

                .. code-block:: python

                    {
                        "period": 1.0914217235792794,
                        "period_err": 1.5396843425628388e-07,
                        "conjunction_time": 2454515.525511919,
                        "conjunction_time_err": 0.00014522035702285087,
                        "period_change_by_epoch": -9.89824903833347e-10,
                        "period_change_by_epoch_err": 7.180878117743563e-11
                    }

        Raises
        ------
        ValueError:
            If model specified is not a valid subclass of :class:`BaseModelEphemeris`, which is either 'linear' or 'quadratic'.
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
        """Calculates the uncertainties of a given linear model when compared to actual data in :class:`TimingData`.
        
        Uses the following equations to calculate the uncertainties between the model data and actual data over epochs:

        - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * E^2)}` for transit observations
        - :math:`\\sigma(\\text{t pred, occ}) = \\sqrt{(\\sigma(T_0)^2 + \\sigma(P)^2 * (\\frac{1}{2} + E)^2)}` for occultation observations
         
        where :math:`\\sigma(T_0)` is conjunction time error, :math:`E` is epoch, and :math:`\\sigma(P)` is period error. 
        
        Parameters
        ----------
        T0_err: numpy.ndarray[float]
            The calculated conjunction time error from a linear model ephemeris.
        P_err: numpy.ndarray[float]
            The calculated period error from a linear model ephemeris.
        
        Returns
        -------
        lin_uncertainties: numpy.ndarray[float]
            A list of uncertainties associated with the model ephemeris data passed in, calculated with the 
            equation above and the :class:`TimingData` epochs.
        """
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(np.sqrt((T0_err**2) + ((self.timing_data.epochs[i]**2)*(P_err**2))))
            elif t_type == 'occ':
                # occultation data
                result.append(np.sqrt((T0_err**2) + (((self.timing_data.epochs[i]+0.5)**2)*(P_err**2))))
        lin_uncertainties = np.array(result)
        return lin_uncertainties
    
    def _calc_quadratic_model_uncertainties(self, T0_err, P_err, dPdE_err):
        """Calculates the uncertainties of a given quadratic model when compared to actual data in :class:`TimingData`.
        
        Uses the following equations to calculate the uncertainties between the model data and actual data over epochs:

        - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * E^2) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))}` for transit observations
        - :math:`\\sigma(\\text{t pred, tra}) = \\sqrt{(\\sigma(T_0)^2 + (\\sigma(P)^2 * (\\frac{1}{2} + E^2)) + (\\frac{1}{4} * \\sigma(\\frac{dP}{dE})^2 * E^4))}` for occultation observations

        where :math:`\\sigma(T_0)` is conjunction time error, :math:`E` is epoch, :math:`\\sigma(P)` is period 
        error, and :math:`\\sigma(\\frac{dP}{dE})` is period change with respect to epoch error.
        
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
        quad_uncertainties: numpy.ndarray[float]
            A list of uncertainties associated with the model ephemeris passed in, calculated with the 
            equation above and the :class:`TimingData` epochs.
        """
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(np.sqrt((T0_err**2) + ((self.timing_data.epochs[i]**2)*(P_err**2)) + ((1/4)*(self.timing_data.epochs[i]**4)*(dPdE_err**2))))
            elif t_type == 'occ':
                # occultation data
                result.append(np.sqrt((T0_err**2) + (((self.timing_data.epochs[i]+0.5)**2)*(P_err**2)) + ((1/4)*(self.timing_data.epochs[i]**4)*(dPdE_err**2))))
        quad_uncertainties = np.array(result)
        return quad_uncertainties
    
    def _calc_linear_ephemeris(self, E, P, T0):
        """Calculates mid-times using parameters from a linear model ephemeris.
        
        Uses the following equations to calculate the mid-time times over each epoch:

        - :math:`T_0 + P*E` for transit observations
        - :math:`(T_0 + \\frac{1}{2}P) + P*E` for occultation observations

        where :math:`T_0` is conjunction time, :math:`P` is period, and :math:`E` is epoch.

        Parameters
        ----------
        E: numpy.ndarray[int]
            The epochs pulled from the :class:`TimingData` object.
        P: float
            The orbital period of the exoplanet as calculated by the linear ephemeris model.
        T0: float
            The conjunction time of the exoplanet as calculated by the linear ephemeris model.

        Returns
        -------
        lin_mid_times: numpy.ndarray[float]
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
        lin_mid_times = np.array(result)
        return lin_mid_times
    
    def _calc_quadratic_ephemeris(self, E, P, T0, dPdE):
        """Calculates mid-times using parameters from a quadratic model ephemeris.

        Uses the following equations to calculate the mid-times over each epoch:

        - :math:`(T_0 + P*E + (\\frac{1}{2} \\frac{dP}{dE} * E^2))` for transit observations
        - :math:`((T_0 + \\frac{1}{2}P) + P*E + (\\frac{1}{2} \\frac{dP}{dE} * E²))` for occultation observations
        
        where :math:`T_0` is conjunction time, :math:`P` is period, :math:`E` is epoch, and 
        :math:`\\frac{dP}{dE}` is period change with respect to epoch.

        Parameters
        ----------
        E: numpy.ndarray[int]
            The epochs pulled from the :class:`TimingData` object.
        P: float
            The orbital period of the exoplanet as calculated by the linear ephemeris model.
        T0: float
            The conjunction time of the exoplanet as calculated by the linear ephemeris model.
        dPdE: float
            The period change with respect to epoch as calculated by the linear ephemeris model.

        Returns
        -------
        quad_mid_times: numpy.ndarray[float]
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
        quad_mid_times = np.array(result)
        return quad_mid_times
    
    def _calc_chi_squared(self, model_mid_times):
        """Calculates the residual chi squared values for the model ephemeris.
        
        Parameters
        ----------
        model_mid_times: numpy.ndarray[float]
            Mid-times calculated from a model ephemeris. This data can be accessed through the ``model_data``
            key from a returned model ephemeris data dictionary. 
        
        Returns
        -------
        chi_squared: float
            The chi-squared value calculated with the equation:
            :math:`\\Sigma(\\frac{(\\text{observed mid-times} - \text{model calculated mid-times})}{\text{observed mid-time uncertainties}^2})`
        """
        # STEP 1: Get observed mid-times
        observed_data = self.timing_data.mid_times
        uncertainties = self.timing_data.mid_time_uncertainties
        # STEP 2: calculate X2 with observed data and model data
        chi_squared = np.sum(((observed_data - model_mid_times)/uncertainties)**2)
        return chi_squared
    
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
    
    def _subtract_plotting_parameters(self, model_mid_times, T0, P, E):
        """Subtracts the first terms to show smaller changes for plotting functions.

        Uses the following equations to subtract the linear terms from the data:

        - :math:`(\\text{model midtime} - T_0 - P*E)` for transit observations
        - :math:`(\\text{model midtime} - T)0 - (\\frac{1}{2}P) - P*E)` for occultation observations
        
        Parameters
        ----------
        model_mid_times : numpy.ndarray[float]
            Mid-times calculated from a model ephemeris. This data can be accessed through the ``model_data``
            key from a returned model ephemeris data dictionary. 
        T0: float
            The conjunction time of the exoplanet as calculated by the linear ephemeris model.
        P: float
            The orbital period of the exoplanet as calculated by the linear ephemeris model.
        E: numpy.ndarray[int]
            The epochs pulled from the :class:`TimingData` object.

        Returns
        -------
        subtracted_params: numpy.ndarray[float]
            A numpy array of linear model parameter values subtracted from mid-times for plotting.
        """
        result = []
        for i, t_type in enumerate(self.timing_data.tra_or_occ):
            if t_type == 'tra':
                # transit data
                result.append(model_mid_times[i] - T0 - (P*E[i]))
            elif t_type == 'occ':
                # occultation data
                result.append(model_mid_times[i] - T0 - (0.5*P) - (P*E[i]))
        subtracted_params = np.array(result)
        return subtracted_params
    
    def _validate(self):
        """Checks that ``timing_data`` is an instance of the :class:`TimingData` object.

        Raises
        ------
        ValueError :
            error raised if ``timing_data`` is not an instance of :class:`TimingData` object.
        """
        if not isinstance(self.timing_data, TimingData):
            raise ValueError("Variable 'timing_data' expected type of object 'TimingData'.")


class BaseModelEphemeris(ABC):
    """Abstract class that defines the structure of different model ephemeris classes."""
    @abstractmethod
    def fit_model(self, x, y, yerr, tra_or_occ):
        """Fits a model ephemeris to timing data.

        Defines the structure for fitting a model (linear or quadratic) to timing data. 
        All subclasses must implement this method.

        Parameters
        ----------
        x: numpy.ndarray[int]
            The epoch data as recieved from the :class:`TimingData` object.
        y: numpy.ndarray[float]
            The mid-time data as recieved from the :class:`TimingData` object.
        yerr: numpy.ndarray[float]
            The mid-time error data as recieved from the :class:`TimingData` object.
        tra_or_occ: numpy.ndarray[str]
            Indicates if each point of data is taken from a transit or an occultation.

        Returns
        -------
        return_data: dict
            A dictionary containing fitted model parameters. 
        """
        pass


class LinearModelEphemeris(BaseModelEphemeris):
    """Subclass of BaseModelEphemeris that implements a linear fit."""
    def lin_fit(self, E, P, T0, tra_or_occ):
        """Calculates a linear function with given data.

        Uses the following equations as a linear function for an LMfit Model.

        - :math:`P * E + T_0` for transit observations 
        - :math:`P * E + (T_0 + \\frac{1}{2}P)` for occultation observations 

        Where :math:`P` is the orbital period, :math:`E` is epoch, and :math:`T_0` is the conjunction time.
        
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
                
            - :math:`P*E + T_0` if the data point is an observed transit
            - :math:`P*E + (T_0 + \\frac{1}{2}*P)` if the data point is an observed occultation
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

        Compares the model ephemeris data to the linear fit created by data in :class:`TimingData` object calculated 
        with ``lin_fit`` method. Then minimizes the difference between the two sets of data. The LMfit Model then 
        returns the parameters of the linear function corresponding to period, conjunction time, and their 
        respective errors. These parameters are returned in a dictionary to the user.

        Parameters
        ----------
        x: numpy.ndarray[int]
            The epoch data as recieved from the :class:`TimingData` object.
        y: numpy.ndarray[float]
            The mid-time data as recieved from the :class:`TimingData` object.
        yerr: numpy.ndarray[float]
            The mid-time error data as recieved from the :class:`TimingData` object.
        tra_or_occ: numpy.ndarray[str]
            Indicates if each point of data is taken from a transit or an occultation.

        Returns
        ------- 
        return_data: dict
            A dictionary of parameters from the fit model ephemeris with the following key value pairs:

            - ``period``: Estimated orbital period of the exoplanet (in units of days)
            - ``period_err``: Uncertainty associated with ``period`` (in units of days)
            - ``conjunction_time``: Time of conjunction of exoplanet transit or occultation (in Julian Date format)
            - ``conjunction_time_err``: Uncertainty associated with conjunction_time

            Example:
                Here is an example of the dictionary returned for a linear model using WASP-12 b data.

                .. code-block:: python

                    {
                        "period": 1.091419640045437,
                        "period_err": 4.210286887084641e-08,
                        "conjunction_time": 2454515.5273143817,
                        "conjunction_time_err": 9.256395817869713e-05
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

        Uses the following equations as a quadratic function for the LMfit Model.
        
        - :math:`(T_0 + (P * E) + (\\frac{1}{2} \\frac{dP}{dE} * (E^2)))` for transit observations
        - :math:`((T_0 + \\frac{1}{2}P) + (P * E) + (\\frac{1}{2} \\frac{dP}{dE} * (E^2)))` for occultation observations 

        Where :math:`E` is the epoch, :math:`P` is the period, :math:`T_0` is the conjunction time, and :math:`\\frac{dP}{dE}` is the change in period over epochs, 
        
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

            - :math:`T_0 + P*E + \\frac{1}{2} \\frac{dP}{dE}*E^2` if the data point is an observed transit
            - :math:`(T_0 + \\frac{1}{2}P) + P*E + \\frac{1}{2} \\frac{dP}{dE}*E^2` if the data point is an observed occultation
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

        Compares the model ephemeris data to the quadratic fit calculated with ``quad_fit`` method. Then minimizes 
        the difference between the two sets of data. The LMfit Model then returns the parameters of the quadratic 
        function corresponding to period, conjunction time, period change by epoch, and their respective errors. 
        These parameters are returned in a dictionary to the user.

        Parameters
        ----------
        x: numpy.ndarray[int]
            The epoch data as recieved from the :class:`TimingData` object.
        y: numpy.ndarray[float]
            The mid-time data as recieved from the :class:`TimingData` object.
        yerr: numpy.ndarray[float]
            The mid-time error data as recieved from the :class:`TimingData` object.
        tra_or_occ: numpy.ndarray[str]
            Indicates if each point of data is taken from a transit or an occultation.

        Returns
        ------- 
        return_data: dict
            A dictionary of parameters from the fit model ephemeris with the following key value pairs:

            - ``period``: Estimated orbital period of the exoplanet (in units of days)
            - ``period_err``: Uncertainty associated with ``period`` (in units of days)
            - ``conjunction_time``: Time of conjunction of exoplanet transit or occultation (in Julian Date format)
            - ``conjunction_time_err``: Uncertainty associated with conjunction_time
            - ``period_change_by_epoch``: The exoplanet period change with respect to epoch (in units of days)
            - ``period_change_by_epoch_err``: The uncertainties associated with ``period_change_by_epoch`` (in units of days)

            Example:
                Here is an example of the dictionary returned for a quadratic model using WASP-12 b data.

                .. code-block:: python

                    {
                        "period": 1.0914217235792794,
                        "period_err": 1.5396843425628388e-07,
                        "conjunction_time": 2454515.525511919,
                        "conjunction_time_err": 0.00014522035702285087,
                        "period_change_by_epoch": -9.89824903833347e-10,
                        "period_change_by_epoch_err": 7.180878117743563e-11
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
        corresponding subclass of :class:`BaseModelEphemeris` and run the ``fit_model`` method to recieve the model 
        ephemeris return data dictionary.
        
        Parameters
        ----------
        model_type: str
            The name of the model ephemeris to create, either 'linear' or 'quadratic'.
        x: numpy.ndarray[int]
            The epoch data as recieved from the :class:`TimingData` object.
        y: numpy.ndarray[float]
            The mid-time data as recieved from the :class:`TimingData` object.
        yerr: numpy.ndarray[float]
            The mid-time error data as recieved from the :class:`TimingData` object.
        tra_or_occ: numpy.ndarray[str]
            Indicates if each point of data is taken from a transit or an occultation.

        Returns
        -------
        Model: dict
            A dictionary of parameters returned by the fit model ephemeris.

            If a linear model was chosen, these parameters are:

            - ``period``: Estimated orbital period of the exoplanet (in units of days)
            - ``period_err``: Uncertainty associated with ``period`` (in units of days)
            - ``conjunction_time``: Time of conjunction of exoplanet transit or occultation (in Julian Date format)
            - ``conjunction_time_err``: Uncertainty associated with conjunction_time

            If a quadratic model was chosen, the same variables are returned, and two additional parameters are included in the dictionary:
                
            - ``period_change_by_epoch``: The exoplanet period change with respect to epoch (in units of days)
            - ``period_change_by_epoch_err``: The uncertainties associated with ``period_change_by_epoch`` (in units of days)

            Example:
                Here is an example of the dictionary returned for a quadratic model using WASP-12 b data.

                .. code-block:: python

                    {
                        "period": 1.0914217235792794,
                        "period_err": 1.5396843425628388e-07,
                        "conjunction_time": 2454515.525511919,
                        "conjunction_time_err": 0.00014522035702285087,
                        "period_change_by_epoch": -9.89824903833347e-10,
                        "period_change_by_epoch_err": 7.180878117743563e-11
                    }

        Raises
        ------
        ValueError:
            If model specified is not a valid subclass of :class:`BaseModelEphemeris`, which is either 'linear' or 'quadratic'.
        """
        models = {
            'linear': LinearModelEphemeris(),
            'quadratic': QuadraticModelEphemeris()
        }
        if model_type not in models:
            raise ValueError(f"Invalid model type: {model_type}")
        model = models[model_type]
        return model.fit_model(x, y, yerr, tra_or_occ)
    

if __name__ == '__main__':
    # STEP 1: Upload datra from file
    bjd_filepath = "../../example_data/wasp12b_tra_occ.csv"
    bjd_no_occs_filepath = "../../example_data/WASP12b_transit_ephemeris.csv"
    isot_filepath = "../../example_data/wasp12b_isot_utc.csv"
    jd_utc_filepath = "../../example_data/wasp12b_jd_utc.csv"
    jd_utc_no_occs_filepath = "../../example_data/wasp12b_jd_utc_tra.csv"
    data = np.genfromtxt(bjd_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    bjd_data_no_occs = np.genfromtxt(bjd_no_occs_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    isot_data = np.genfromtxt(isot_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    jd_utc_data = np.genfromtxt(jd_utc_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    jd_utc_no_occs_data = np.genfromtxt(jd_utc_no_occs_filepath, delimiter=',', names=True, dtype=None, encoding=None)
    # STEP 2: Break data up into epochs, mid-times, and error
    # STEP 2.5 (Optional): Make sure the epochs are integers and not floats
    tra_or_occs = data["tra_or_occ"] # Base tra_or_occs
    epochs = data["epoch"].astype('int') # Epochs with tra_or_occs
    epochs_no_occs = bjd_data_no_occs["epoch"].astype('int') # Epochs with ONLY tra
    isot_mid_times = isot_data["transit_time"] # ISOT mid times
    jd_utc_times = jd_utc_data["transit_time"] # JD UTC mid times
    jd_utc_time_errs = jd_utc_data["sigma_transit_time"] # JD UTC mid time errs
    jd_utc_times_no_occs = jd_utc_no_occs_data["transit_time"] # JD UTC mid times ONLY tra
    jd_utc_time_errs_no_occs = jd_utc_no_occs_data["sigma_transit_time"] # JD UTC mid time errs ONLY tra
    bjd_mid_times = data["transit_time"] # BJD mid times
    bjd_mid_time_errs = data["sigma_transit_time"] # BJD mid time errs
    bjd_mid_times_no_occs = bjd_data_no_occs["transit_time"]
    bjd_mid_time_errs_no_occs = bjd_data_no_occs["sigma_transit_time"]
    # print(f"epochs: {list(epochs)}")
    # print(f"mid_times: {list(mid_times)}")
    # print(f"mid_time_errs: {list(mid_time_errs)}")
    # print(f"tra_or_occ: {list(tra_or_occs)}")
    # STEP 3: Create new transit times object with above data
    """NOTE: ISOT (NO UNCERTAINTIES)"""
    # times_obj1 = TimingData('isot', epochs, isot_mid_times, tra_or_occ=tra_or_occs, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    """NOTE: JD UTC"""
    # times_obj1 = TimingData('jd', epochs, jd_utc_times, jd_utc_time_errs, tra_or_occ=tra_or_occs, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    """NOTE: JD UTC NO UNCERTAINTIES"""
    # times_obj1 = TimingData('jd', epochs, jd_utc_times, tra_or_occ=tra_or_occs, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    """NOTE: JD UTC NO UNCERTAINTIES NO TRA_OR_OCC"""
    # times_obj1 = TimingData('jd', epochs_no_occs, jd_utc_times_no_occs, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)
    """NOTE: JD TDB (BJD) NO TRA_OR_OCC"""
    # times_obj1 = TimingData('jd', epochs_no_occs, bjd_mid_times_no_occs, bjd_mid_time_errs_no_occs, time_scale='tdb')
    """NOTE: JD TDB (BJD) ALL INFO"""
    times_obj1 = TimingData('jd', epochs, bjd_mid_times, bjd_mid_time_errs, time_scale='tdb', tra_or_occ=tra_or_occs)
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
    ephemeris_obj1.plot_model_ephemeris(linear_model_data, save_plot=True, fname="../../wasp12b_graphs/lin_model")
    ephemeris_obj1.plot_model_ephemeris(quad_model_data, save_plot=True, fname="../../wasp12b_graphs/quad_model")

    # STEP 7: Uncertainties plot
    # ephemeris_obj1.plot_timing_uncertainties(linear_model_data, save_plot=False)
    # ephemeris_obj1.plot_timing_uncertainties(quad_model_data, save_plot=False)
    ephemeris_obj1.plot_timing_uncertainties(linear_model_data, save_plot=True, fname="../../wasp12b_graphs/lin_unc")
    ephemeris_obj1.plot_timing_uncertainties(quad_model_data, save_plot=True, fname="../../wasp12b_graphs/quad_unc")
    
    # STEP 8: O-C Plot
    # ephemeris_obj1.plot_oc_plot(save_plot=False)
    ephemeris_obj1.plot_oc_plot(save_plot=True, fname="../../wasp12b_graphs/oc_plot")

    # STEP 9: Running delta BIC plot
    # ephemeris_obj1.plot_running_delta_bic(save_plot=False)
    ephemeris_obj1.plot_running_delta_bic(save_plot=True, fname="../../wasp12b_graphs/running_delta_bic")

