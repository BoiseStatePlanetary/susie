# The Susie Python Package
A package for exoplanet transit decay calculations and visualizations.

![Susie Superpig Cartoon Image](http://www.astrojack.com/wp-content/uploads/2013/12/susie-1024x748.png)

## Statement of need
The authors should clearly state what problems the software is designed to solve, who the target audience is, and its relation to other work.

## Installation instructions
To download this package, use:
`pip install susie`

This package uses numpy, scipy, matplotlib, and astropy software. These packages will be downloaded with the pip install command.

## Example usage
There are two main objects to use in this package:

`Ephemeris` and `TransitTimes`.

The ephemeris object contains methods for fitting transit data to model ephemerides to perform tidal decay calculations and visualizations. The transit data is inputted into the TransitTimes object. 

The user must first instantiate a TransitTimes object. The transit times object takes the following attributes:
 - `time_format` (str): 
 - `epochs` (np.array of ints):
 - `mid_transit_times` (np.array of floats):
 - `mid_transit_times_uncertainties` (np.array of floats or None) 
 - `time_scale` (str):
 - `object_ra` (float):
 - `object_dec` (float):
 - `observatory_lon` (float):
 - `observatory_lat` (float):

The following are valid instantiations of the TransitTimes object:

`transit_times = TransitTimes()`
`transit_times = TransitTimes()`
`transit_times = TransitTimes()`
`transit_times = TransitTimes()`
`transit_times = TransitTimes()`

Once the TransitTimes object is instantiated, it can be used to instantiate the Ephemeris object. The following is a valid instantiation of the Ephemeris object.

`ephemeris1 = Ephemeris(transit_times)`

Once the ephemeris object is successfully created, the user can run the following methods:

``
``
``
``
``
``

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