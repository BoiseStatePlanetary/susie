# The Susie Python Package
A package for exoplanet transit decay calculations and visualizations.

![Susie Superpig Cartoon Image](http://www.astrojack.com/wp-content/uploads/2013/12/susie-1024x748.png)

## Statement of need
The authors should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work.

## Installation instructions
To download this package, use:
`pip install susie`

This package uses numpy, scipy, matplotlib, and astropy software. These packages will be downloaded with the pip install command.

## Objects

### TransitTimes
Represents transit midpoint data over time. Holds data to be accessed by Ephemeris class.

**Arguments:**
 - **`time_format`** (str): An abbreviation of the data's timing system. Abbreviations for systems can be found on [Astropy's Time documentation](https://docs.astropy.org/en/stable/time/#id3).
 - `epochs` (numpy.ndarray[int]): List of reference points for transit observations represented in the transit times data.
 - `mid_transit_times` (numpy.ndarray[float]): List of observed transit midpoints corresponding with epochs.
 - `mid_transit_times_uncertainties` (Optional[numpy.ndarray[float]]): List of uncertainties corresponding with transit midpoints. If given None, will be replaced with array of 1's with same shape as `mid_transit_times`.
 - `time_scale` (Optional[str]): An abbreviation of the data's timing scale. Abbreviations for scales can be found on [Astropy's Time documentation](https://docs.astropy.org/en/stable/time/#id6).
 - `object_ra` (Optional[float]): The right ascension in degrees of observed object represented by data.
 - `object_dec` (Optional[float]): The declination in degrees of observed object represented by data.
 - `observatory_lon` (Optional[float]): The longitude in degrees of observatory data was collected from.
 - `observatory_lat` (Optional[float]): The latitude in degrees of observatory data was collected from.


### Ephemeris
Represents the model ephemeris using transit midpoint data over epochs.

**Arguments:**
 - `transit_times` (TransitTimes): A successfully instantiated TransitTimes object holding epochs, mid transit times, and uncertainties.

**Methods:**
`get_model_ephemeris`
    Fits the transit data to a specified model using [SciPy's `curve_fit` function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html).
    **Parameters**:
        `model_type` (str): Either 'linear' or 'quadratic'. Represents the type of ephemeris to fit the data to.
    **Returns**:
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

`get_ephemeris_uncertainties`
    Calculates the uncertainties of a specific model data when compared to the actual data. Uses the equation $σ_t_tra^pred = \sqrt{σ_T0^2 + (E^2 * σ_P^2)}$ for linear models and $σ_t_tra^pred = \sqrt{σ_T0^2 + (E^2 * σ_P^2) + (1/4 * σ_dP/dE^2 * E^4)}$ for quadratic models (where $σ_T0 = conjunction time error, E = epoch, σ_P = period error, and σ_dP/dE = period change by epoch error$).
    **Parameters**: 
        `model_data_dict` (dict): A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
    **Returns**:
        A list of uncertainties associated with the model ephemeris passed in, calculated with the equations above and the passed in model data.

`calc_bic`
    Calculates the BIC value for a given model ephemeris. Uses the equation $ 𝜒^2 + (k * \log(N))$ where $𝜒^2=\sum ((observed mid transit times - model ephemeris mid transit times)/observed mid transit time uncertainties)^2$, k=number of fit parameters (2 for linear models, 3 for quadratic models), and N=total number of data points.
    **Parameters**:
        `model_data_dict` (dict): A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
    **Returns**:
        A float value representing the BIC value for this model ephemeris.
    
`calc_delta_bic`
    Calculates the ΔBIC value between linear and quadratic model ephemerides using the given transit data. 
    **Returns**:
        A float value representing the ΔBIC value for this transit data.
    
`plot_model_ephemeris`
    Returns a MatplotLib scatter plot showing predicted mid transit times from the model ephemeris over epochs.
    **Parameters**:
        - `model_data_dict` (dict): A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
        - `save_plot` (bool): If True, will save the plot as a figure.
        - `save_filepath` (Optional[str]): The path used to save the plot if `save_plot` is True.
    **Returns**:
        A MatplotLib plot of epochs vs. model predicted mid-transit times.
    
`plot_timing_uncertainties`
    **Parameters**:
        - `model_data_dict` (dict): A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
        - `save_plot` (bool): If True, will save the plot as a figure.
        - `save_filepath` (Optional[str]): The path used to save the plot if `save_plot` is True.
    **Returns**:
        A MatplotLib plot of timing uncertainties.
    
`plot_oc_plot`
    **Parameters**:
        - `save_plot` (bool): If True, will save the plot as a figure.
        - `save_filepath` (Optional[str]): The path used to save the plot if `save_plot` is True.
    **Returns**:
        A MatplotLib plot of observed vs. calculated values of mid transit times for linear and quadratic model ephemerides over epochs.
    
`plot_running_delta_bic`
    **Parameters**:
        - `save_plot` (bool): If True, will save the plot as a figure.
        - `save_filepath` (Optional[str]): The path used to save the plot if `save_plot` is True.
    **Returns**:
        A MatplotLib scatter plot of epochs vs. ΔBIC for each epoch.
    

## Example usage
There are two main objects to use in this package:

`Ephemeris` and `TransitTimes`.

The ephemeris object contains methods for fitting transit data to model ephemerides to perform tidal decay calculations and visualizations. The transit data is inputted into the TransitTimes object. 

The user must first instantiate a TransitTimes object. Once the TransitTimes object is instantiated, it can be used to instantiate the Ephemeris object. Some examples in instantiating and using the TransitTimes and Ephemeris objects are below.

### Load In and Process Data
First need to get data. This is an example of hard-coded values.
```
# STEP 1: Load in data
epoch_data = [-1640.0, -1346.0, -1342.0, -1067.0, -1061.0, -1046.0, -1038.0, -1004.0, -1003.0, -985.0, -963.0, -743.0, -739.0, -729.0, -728.0, -721.0, -699.0, -699.0, -677.0, -655.0, -648.0, -646.0, -645.0, -643.0, -625.0]
mid_transit_time_data = [2454515.525, 2454836.403, 2454840.769, 2455140.91, 2455147.459, 2455163.831, 2455172.561, 2455209.669, 2455210.762, 2455230.407, 2455254.419, 2455494.53, 2455498.896, 2455509.81, 2455510.902, 2455518.541, 2455542.552, 2455542.553, 2455566.564, 2455590.576, 2455598.216, 2455600.398, 2455601.49, 2455603.673, 2455623.318]
mid_transit_times_uncertainties_data = [0.00043, 0.00028, 0.00062, 0.00042, 0.00043, 0.00032, 0.00036, 0.00046, 0.00041, 0.00019, 0.00043, 0.00072, 0.00079, 0.00037, 0.00031, 0.0004, 0.0004, 0.00028, 0.00028, 0.00068, 0.00035, 0.00029, 0.00024, 0.00029, 0.00039]
# STEP 2: Normalize data to start from 0
epochs = epoch_data - np.min(epoch_data)
mid_transit_times = mid_transit_time_data - np.min(mid_transit_time_data)
mid_transit_times_err = mid_transit_times_uncertainties_data
# STEP 2.5 (Optional): Make sure the epochs are integers and not floats
epochs = epochs.astype('int')
```

### Instantiate TransitTimes
Then, we can instantiate our TransitTimes object. There are a few ways we can do this, some common ways are included below...

With times already corrected for barycentric light travel times:
    `transit_times = TransitTimes('jd', epochs, mid_transit_times, mid_transit_times_err, time_scale='tdb')`

With times already corrected for barycentric light travel times, no uncertainties given:
    `transit_times = TransitTimes('jd', epochs, mid_transit_times, time_scale='tdb')`

With times NOT corrected for barycentric light travel times:
    `transit_times = TransitTimes('jd', epochs, mid_transit_times, mid_transit_times_err, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)`

With times NOT corrected for barycentric light travel times and no observatory coordinates given:
    `transit_times = TransitTimes('jd', epochs, mid_transit_times, mid_transit_times_err, object_ra=97.64, object_dec=29.67)`

### Instantiate Ephemeris
Now, we can instantiate our Ephemeris class using our TransitTimes object and perform some function calls.

`ephemeris = Ephemeris(transit_times)`

The authors should include examples of how to use the software (ideally to solve real-world analysis problems).
TODO: This will probably somehow be pulled from a notebook, we can also include some graphs and stuff.

## API documentation
Reviewers should check that the software API is documented to a suitable level.
TODO: <\insert link to documentation>

## Community guidelines
To report bugs or create pull requests, please visit the Github repository [here]().
There should be clear guidelines for third-parties wishing to:
Contribute to the software
Report issues or problems with the software
Seek support

## Links
[Test PyPi](https://test.pypi.org/project/Susie/0.0.1/)