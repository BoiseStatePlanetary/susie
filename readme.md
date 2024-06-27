# The Susie Python Package
[![Documentation Status](https://readthedocs.org/projects/susie/badge/?version=latest)](https://susie.readthedocs.io/en/latest/index.html)
![Coverage](https://raw.githubusercontent.com/BoiseStatePlanetary/susie/5475f3a48a346551fb6bc3910055f09a854e8a89/readme_imgs/coverage.svg)
[![Powered by Astropy](https://img.shields.io/badge/powered%20by-AstroPy-orange?style=flat)](https://www.astropy.org/)

A package for exoplanet orbital decay calculations and visualizations.

[![Susie Superpig Cartoon Image](https://github.com/BoiseStatePlanetary/susie/blob/main/readme_imgs/SP_Susie.png?raw=true)](https://susie.readthedocs.io/en/latest/index.html)

## Statement of need
Hot Jupiters are planets outside of our solar system that orbit extremely fast and close to their star. Due to their proximity, hot Jupiters experience gravitational forces from their stars, and are large enough to also exhibit gravitational forces ON their stars. These forces are called tidal interactions, and are similar to the tidal interactions we see between our moon and Earth. The tidal interactions between a hot Jupiter and its star gradually pull the hot Jupiter closer to the star. As the hot Jupiter gets closer to the star, the planet completes a full orbit around its star faster. Therefore, the time it takes for the planet to complete an orbit—which we call the orbital period—decreases. We can find these hot Jupiters by looking for these decreasing orbital periods using observations of the orbits. 

![Illustration of an exoplanet transit. As the planet passes in front of its host star, the total amount of light coming from the star drops, as seen in the U-shaped curve. The time between the middle of each transit curve equals the planet’s orbital period.](https://github.com/BoiseStatePlanetary/susie/blob/main/readme_imgs/Transit_Illustration.png?raw=true)

A common way of observing an exoplanet's orbit is by observing a transit. An exoplanet 'transits' its star when it passes in front of the star, blocking out some of the light. We can collect the amount of light coming from the star over time, and when the exoplanet transits the star, we see the amount of light decrease. By looking at a graph of amount of star light vs. time, such as the graph above, we can record the duration of the transit and find the moment an exoplanet is exactly halfway through the transit, called the transit mid-time. By comparing one mid time to the next, we can measure an orbital period. By observing transits over time, we can find changes in mid-times and therefore the changes in orbital periods that would indicate hot Jupiters experiencing tidal interactions with their stars. However, these changes in orbital period are incredibly small, with timing variations of milliseconds per Earth year, making an unguided search for these systems inefficient. Our package helps to streamline this process.

## Installation instructions
To download this package, use:
`pip install susie`

This package uses numpy, scipy, matplotlib, and astropy software. These packages will be downloaded with the pip install command.

## Objects

### TimingData
Represents a collection of mid transit and/or occultation times. Holds data to be accessed by Ephemeris class.

**Arguments:**
 - `time_format` (str): An abbreviation of the data's timing system. Abbreviations for systems can be found on [Astropy's Time documentation](https://docs.astropy.org/en/stable/time/#id3).
 - `epochs` (numpy.ndarray[int]): List of reference points for transit observations represented in the transit times data.
 - `mid_times` (numpy.ndarray[float]): List of observed transit and/or occultation mid-times corresponding with epochs.
 - `mid_time_uncertainties` (Optional[numpy.ndarray[float]]): List of uncertainties corresponding with mid-times. If given None, will be replaced with array of 1's with same shape as `mid_times`.
 - `tra_or_occ` (Optional[numpy.ndarray[str]]): A list of strings corresponding to each point of mid-time data to indicate whether the data came from a transit or an occultations. Will default to transits if not given.
 - `time_scale` (Optional[str]): An abbreviation of the data's timing scale. Abbreviations for scales can be found on [Astropy's Time documentation](https://docs.astropy.org/en/stable/time/#id6).
 - `object_ra` (Optional[float]): The right ascension in degrees of observed object represented by data.
 - `object_dec` (Optional[float]): The declination in degrees of observed object represented by data.
 - `observatory_lon` (Optional[float]): The longitude in degrees of observatory data was collected from.
 - `observatory_lat` (Optional[float]): The latitude in degrees of observatory data was collected from.


### Ephemeris
Represents the model ephemeris using timing midpoint data over epochs.

**Arguments:**
 - `timing_data` (TimingData): A successfully instantiated TimingData object holding epochs, mid times, and uncertainties.

**Methods:**
 - `get_model_ephemeris`
    Fits the timing data to a specified model using an [LMfit Model](https://pypi.org/project/lmfit/).
     - **Parameters**:
        `model_type` (str): Either 'linear' or 'quadratic'. Represents the type of ephemeris to fit the data to.
     - **Returns**:
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

 - `get_ephemeris_uncertainties`
    Calculates the uncertainties of a specific model data when compared to the actual data. Uses the equation $σ_{t_tra^{pred}} = \sqrt{σ_T0^2 + (E^2 * σ_P^2)}$ for linear models and $σ_{t_tra^{pred}} = \sqrt{σ_T0^2 + (E^2 * σ_P^2) + (1/4 * σ_{\frac{dP}{dE}}^2 * E^4)}$ for quadratic models (where $σ_T0$ = conjunction time error, E = epoch, $σ_P$ = period error, and $σ_{\frac{dP}{dE}}$ = period change by epoch error).
     - **Parameters**: 
        `model_data_dict` (dict): A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
     - **Returns**:
        A list of uncertainties associated with the model ephemeris passed in, calculated with the equations above and the passed in model data.

 - `calc_bic`
    Calculates the BIC value for a given model ephemeris. Uses the equation $\chi^2 + (k * \log(N))$ where $𝜒^2=\sum(\frac{(\texttt{observed mid times} - \texttt{model ephemeris mid times})}{\texttt{observed mid time uncertainties}})^2$, k = number of fit parameters (2 for linear models, 3 for quadratic models), and N = total number of data points.
     - **Parameters**:
        `model_data_dict` (dict): A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
     - **Returns**:
        A float value representing the BIC value for this model ephemeris.
    
 - `calc_delta_bic`
    Calculates the ΔBIC value between linear and quadratic model ephemerides using the given transit data. 
     - **Returns**:
        A float value representing the ΔBIC value for this transit data.
    
 - `plot_model_ephemeris`
    Returns a MatplotLib scatter plot showing predicted mid transit times from the model ephemeris over epochs.
     - **Parameters**:
        - `model_data_dict` (dict): A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
        - `save_plot` (bool): If True, will save the plot as a figure.
        - `save_filepath` (Optional[str]): The path used to save the plot if `save_plot` is True.
     - **Returns**:
        A MatplotLib plot of epochs vs. model predicted mid-transit times.
    
 - `plot_timing_uncertainties`
     - **Parameters**:
        - `model_data_dict` (dict): A dictionary of model ephemeris parameters recieved from `Ephemeris.get_model_ephemeris`.
        - `save_plot` (bool): If True, will save the plot as a figure.
        - `save_filepath` (Optional[str]): The path used to save the plot if `save_plot` is True.
     - **Returns**:
        A MatplotLib plot of timing uncertainties.
    
 - `plot_oc_plot`
     - **Parameters**:
        - `save_plot` (bool): If True, will save the plot as a figure.
        - `save_filepath` (Optional[str]): The path used to save the plot if `save_plot` is True.
     - **Returns**:
        A MatplotLib plot of observed vs. calculated values of mid transit times for linear and quadratic model ephemerides over epochs.
    
 - `plot_running_delta_bic`
     - **Parameters**:
        - `save_plot` (bool): If True, will save the plot as a figure.
        - `save_filepath` (Optional[str]): The path used to save the plot if `save_plot` is True.
     - **Returns**:
        A MatplotLib scatter plot of epochs vs. ΔBIC for each epoch.
    

## Example usage
There are two main objects to use in this package:

`Ephemeris` and `TimingData`.

The ephemeris object contains methods for fitting timing data to model ephemerides to perform tidal decay calculations and visualizations. The timing data is inputted into the TimingData object. 

The user must first instantiate a TimingData object. Once the TimingData object is instantiated, it can be used to instantiate the Ephemeris object. Some examples in instantiating and using the TimingData and Ephemeris objects are below.

### Load In and Process Data
First, you need to get data. This is an example of hard-coded values. More examples, including an example with both transits and occultations, are included under the `example_data` file in this repository.
```
# STEP 1: Load in data
epochs = [-1640, -1408, -1404, -1346, -1342, -1067, -1061, -1046, -1038, -1018, -1011, -1004, -1003, -985, -963, -743, -739, -729, -728, -722, -721, -699, -699, -677, -668, -658, -655, -648, -646, -645, -643, -625, -393, -383, -382, -380, -368, -362, -353, -350, -350, -329, -329, -328, -327, -324, -323, -317, -316, -307, -306, -296, -295, -294, -293, -283, -275, -274, -55, -51, -29, -21, -19, -7, -3, 0, 13, 21, 22, 274, 275, 276, 277, 297, 298, 305, 308, 320, 324, 327, 328, 329, 338, 341, 351, 356, 365, 372, 379, 381, 382, 624, 646, 648, 678, 679, 691, 692, 698, 699, 731, 953, 994, 995, 1027, 1028, 1050, 1252, 1270, 1281, 1284, 1302, 1303, 1338, 1339, 1341, 1344, 1345, 1348, 1352, 1352, 1357, 1359, 1370, 1377, 1378, 1379, 1397, 1577, 1599, 1608, 1620, 1621, 1624, 1666, 1667, 1674, 1676, 1678, 1695, 1695, 1696, 1699, 1701, 1701, 1702, 1702, 1705, 1716, 1930, 1985, 1997, 2006, 2010, 2014, 2017, 2018, 2026]
mid_times = [2454515.52496, 2454769.2819, 2454773.6481, 2454836.4034, 2454840.76893, 2455140.90981, 2455147.45861, 2455163.83061, 2455172.56138, 2455194.9344, 2455202.57625, 2455209.66895, 2455210.76151, 2455230.40669, 2455254.41871, 2455494.52999, 2455498.8959, 2455509.80971, 2455510.90218, 2455517.99514, 2455518.5407, 2455542.5521, 2455542.55273, 2455566.56385, 2455576.932, 2455587.8473, 2455590.57561, 2455598.21552, 2455600.398, 2455601.4901, 2455603.67261, 2455623.31829, 2455876.52786, 2455887.44198, 2455888.5334, 2455890.71635, 2455903.81357, 2455910.909, 2455920.18422, 2455923.4585, 2455924.0047, 2455946.37823, 2455946.9229, 2455947.47015, 2455948.56112, 2455951.83534, 2455952.9272, 2455959.47543, 2455960.56686, 2455970.38941, 2455971.48111, 2455982.39509, 2455983.48695, 2455984.57797, 2455985.66975, 2455996.58378, 2456005.31533, 2456006.40637, 2456245.42729, 2456249.79404, 2456273.80514, 2456282.53584, 2456284.71857, 2456297.81605, 2456302.18179, 2456305.45536, 2456319.64424, 2456328.37556, 2456329.46733, 2456604.50489, 2456605.59624, 2456606.6876, 2456607.77938, 2456629.60726, 2456630.69917, 2456638.88589, 2456642.15907, 2456654.71047, 2456659.07598, 2456662.35014, 2456663.44136, 2456664.53256, 2456674.3556, 2456677.63039, 2456688.54384, 2456694.00161, 2456703.82417, 2456711.46415, 2456719.10428, 2456721.28692, 2456722.37807, 2456986.50195, 2457010.51298, 2457012.69617, 2457045.43831, 2457046.53019, 2457059.62713, 2457060.71839, 2457067.26715, 2457068.35834, 2457103.28423, 2457345.57867, 2457390.32708, 2457391.41818, 2457426.34324, 2457427.43496, 2457451.44617, 2457671.91324, 2457691.55888, 2457703.56388, 2457706.83791, 2457726.484, 2457727.57547, 2457765.77515, 2457766.86633, 2457769.59573, 2457772.32407, 2457773.41517, 2457776.68869, 2457781.05418, 2457781.05566, 2457786.5121, 2457788.69464, 2457800.69978, 2457808.3402, 2457809.4319, 2457810.52327, 2457830.71389, 2458026.62368, 2458050.63519, 2458060.4587, 2458073.55509, 2458074.64651, 2458077.92107, 2458123.76011, 2458124.85183, 2458132.49121, 2458134.67471, 2458136.8576, 2458155.4104, 2458155.41152, 2458156.50267, 2458159.77773, 2458161.95991, 2458161.95964, 2458163.05125, 2458163.05089, 2458166.32575, 2458178.33104, 2458411.89495, 2458471.92257, 2458485.56424, 2458494.8427, 2458499.75572, 2458504.11988, 2458506.84758, 2458508.48459, 2458517.21641]
mid_time_uncertainties = [0.00043, 0.0008, 0.0006, 0.00028, 0.00062, 0.00042, 0.00043, 0.00032, 0.00036, 0.001, 0.0022, 0.00046, 0.00041, 0.00019, 0.00043, 0.00072, 0.00079, 0.00037, 0.00031, 0.00118, 0.0004, 0.0004, 0.00028, 0.00028, 0.0009, 0.0017, 0.00068, 0.00035, 0.00029, 0.00024, 0.00029, 0.00039, 0.00027, 0.00021, 0.00027, 0.00024, 0.00032, 0.0013, 0.00031, 0.00022, 0.0021, 0.00018, 0.0018, 0.00017, 0.00033, 0.00011, 0.0001, 0.00017, 0.00032, 0.00039, 0.00035, 0.00034, 0.00035, 0.00032, 0.00042, 0.00037, 0.00037, 0.00031, 0.00033, 0.00039, 0.0003, 0.0003, 0.0003, 0.0003, 0.00046, 0.00024, 0.00038, 0.00027, 0.00029, 0.00021, 0.0003, 0.00033, 0.00071, 0.00019, 0.00043, 0.0011, 0.00141, 0.00034, 0.00034, 0.00019, 0.00019, 0.00031, 0.00028, 0.00032, 0.0004, 0.00029, 0.00029, 0.00025, 0.00034, 0.00034, 0.00046, 0.00043, 0.00039, 0.00049, 0.00046, 0.00049, 0.00035, 0.00036, 0.00022, 0.0002, 0.00031, 0.00042, 0.00033, 0.00033, 0.00055, 0.00023, 0.00021, 0.00035, 0.00025, 0.00034, 0.00037, 0.00028, 0.00023, 0.00028, 0.00039, 0.00136, 0.00024, 0.00022, 0.00029, 0.00043, 0.00036, 0.00026, 0.00048, 0.00032, 0.0004, 0.00018, 0.00021, 0.0011, 0.00056, 0.00023, 0.0003, 0.00022, 0.00034, 0.00028, 0.00027, 0.00035, 0.00031, 0.00032, 0.00033, 0.0005, 0.00031, 0.00032, 0.00091, 0.00035, 0.00026, 0.00021, 0.00034, 0.00034, 0.00038, 0.0004, 0.00026, 0.0014, 0.0003, 0.00077, 0.00087, 0.00044, 0.00091, 0.00074]
tra_or_occs = ['tra', 'occ', 'occ', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'occ', 'occ', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'occ', 'tra', 'tra', 'tra', 'tra', 'occ', 'occ', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'occ', 'tra', 'tra', 'occ', 'tra', 'occ', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'occ', 'occ', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'occ', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'occ', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'tra', 'occ', 'tra', 'occ', 'occ', 'tra', 'occ', 'occ']
```
This data is pulled from [Yee et. al.](https://iopscience.iop.org/article/10.3847/2041-8213/ab5c16#apjlab5c16t5) and can be downloaded in a CSV format [here](https://content.cld.iop.org/journals/2041-8205/888/1/L5/revision1/apjlab5c16t5_mrt.txt). The data represents observations of the hot Jupiter WASP 12-b.

### Instantiate TimingData
Then, we can instantiate our TimingData object. There are a few ways we can do this, some common ways are included below...

With times already corrected for barycentric light travel times:
    `timing_data = TimingData('jd', epochs, mid_times, mid_time_errs, time_scale='tdb', tra_or_occ=tra_or_occs)`

With times already corrected for barycentric light travel times, no uncertainties given:
    `timing_data = TimingData('jd', epochs, mid_times, time_scale='tdb', tra_or_occ=tra_or_occs)`

With times already corrected for barycentric light travel times, no list of transits and occultations given:   
    `timing_data = TimingData('jd', epochs, mid_times, mid_time_errs, time_scale='tdb')`

With times NOT corrected for barycentric light travel times:
    `timing_data = TimingData('jd', epochs, mid_times, mid_time_errs, tra_or_occ=tra_or_occs, object_ra=97.64, object_dec=29.67, observatory_lat=43.60, observatory_lon=-116.21)`

With times NOT corrected for barycentric light travel times and no observatory coordinates given:
    `timing_data = TimingData('jd', epochs, mid_times, mid_time_errs, tra_or_occ=tra_or_occs, object_ra=97.64, object_dec=29.67)`

There are many different ways to import your data, and you can use any mix of options mentioned above. Ideally, include as much information as you can. More examples are included on our [ReadTheDocs page](https://susie.readthedocs.io/en/latest).

### Instantiate Ephemeris
Now, we can instantiate our Ephemeris class using our TimingData object and perform some function calls.

`ephemeris = Ephemeris(timing_data)`

To start, you can return the $\Delta$ BIC value, which will give you a good idea if your system is exhibiting tidal decay (a large positive value) or not (a negative value). To get the value of $\Delta$ BIC for your system, call the function:

`delta_bic = ephemeris.calc_delta_bic()`

You can also visualize both the linear and quadratic model fits and compare these fits to the actual data with an observed minus calculated plot. To get this plot, call the function:

`ephemeris_obj1.plot_oc_plot()`

Here is an example of an OC plot returned by the data above.

![An Observed minus Caluclated plot for the hot Jupiter WASP 12-b.](https://github.com/BoiseStatePlanetary/susie/blob/main/readme_imgs/Figure_1.png?raw=true)

<small>NOTE: You also have the option to save this plot by setting `save_plot` to True and including a `file_path` parameter. This option is available for ALL plotting methods.</small>

For more details on methods, data, visualizations, and examples, check out our [ReadTheDocs page](https://susie.readthedocs.io/en/latest)!

## API documentation
Documentation can be accessed on our public [ReadTheDocs page](https://susie.readthedocs.io/en/latest).

## Community guidelines
To report bugs or create pull requests, please visit the Github repository [here](https://github.com/BoiseStatePlanetary/susie).

Copyright 2023 Boise State University

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Links
[Test PyPi](https://test.pypi.org/project/Susie/0.0.1/)