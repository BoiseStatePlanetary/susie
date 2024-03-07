.. Susie documentation master file, created by
   sphinx-quickstart on Sun Oct 15 12:14:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Susie's documentation!
=================================

Susie is a package which determines if exoplanets are good candidates for tidal decay. It does this by examining the sign (and magnitude??) of the :math:`\Delta BIC` for linear and quadratic models of the same transit data.

:ref:`transit_times.py <transit_times_label>` corrects and holds the data to be accessed by the Ephemeris class.

:ref:`ephemeris.py <ephemeris_label>` uses the transit midpoint data and epochs held by the Transit Times class to calculate the :math:`\Delta BIC` value, and then creates a visual output using MatPlotLib comparing a linear and quadratic fit. 

.. toctree::
   :caption: Contents:

   susie
   usage
   test

.. note::
   This project is under active development.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


The developers were supported by a TESS Guest Investigator Cycle-5 grant (NNH21ZDA001N), and 
an Exoplanets Research grant (NNH21ZDA001N).