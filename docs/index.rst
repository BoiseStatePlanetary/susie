.. Susie documentation master file, created by
   sphinx-quickstart on Sun Oct 15 12:14:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Susie's documentation!
=================================

Susie is a Python package for modeling the orbital evolution of exoplanets using transit and occultation timing data. It helps detect and characterize tidal orbital decay and apsidal precession, offering statistical tools to compare models and infer the physical processes shaping close-in planetary systems—linking timing data to insights on tidal dissipation, planetary interiors, and dynamical interactions.

.. image:: images/SP_Susie.png
   :alt: The Susie Exoplanets Group Logo

About
-----

**Susie** is a Python package for fitting models to exoplanet transit and occultation timing data in order to investigate orbital evolution. It is designed to detect deviations from constant orbital motion, such as tidal orbital decay or apsidal precession, by fitting the data to different timing models and comparing their statistical likelihoods.

Susie supports three models:

- **Linear**: constant-period orbit (no orbital evolution)
- **Quadratic**: changing orbital period (e.g., due to tidal decay)
- **Sinusoidal**: precession of the orbit (e.g., due to apsidal motion)

The workflow is as follows:

1. Fit your timing data to one or more of the models (linear, quadratic, or sinusoidal).
2. Evaluate the quality of the model fit using the Bayesian Information Criterion (BIC).
3. Compare models using ΔBIC (difference in BIC values) to determine which model best explains the data.
4. Visualize model parameters and fit residuals to better interpret the orbital behavior.

A lower BIC indicates a better-fitting model. ΔBIC is computed as the BIC of one model minus the BIC of another. While it’s common to compare a non-linear model (quadratic or sinusoidal) against a linear one to test for non-constant orbits, any pair of models can be compared. A positive ΔBIC value indicates that the second model better explains the data (i.e., it has a lower BIC).

Susie allows you to:

- Fit individual models and examine their parameter estimates and fit diagnostics.
- Automatically fit multiple models and return a summary of BIC values and pairwise ΔBIC comparisons.
- Generate clear visualizations to assess model performance and potential orbital changes.

To get started, load your timing data using the ``timing_data.py`` object, then pass it to ``ephemeris.py`` to perform model fitting, comparison, and visualization.

Planned features include:

- Integration with `Astroplan <https://astroplan.readthedocs.io/en/stable/index.html>`_ to plan future transit observations.
- Improved parameter estimation using the `emcee <https://emcee.readthedocs.io/en/stable/>`_ MCMC sampler.
- Extended support for distinguishing precession from decay in ambiguous timing signals.

.. note::
   This project is under active development.


Quickstart
----------

#. Install the latest released version from the Python Package Index `PyPi <https://pypi.org/project/susie/>`_:

   .. code-block:: shell

      pip install susie

#. Include at the top of your Python file or notebook:

   .. code-block:: python

      import susie

   or 

   .. code-block:: python

      from susie.timing_data import TimingData
      from susie.ephemeris import Ephemeris


Documentation
-------------
.. toctree::

   susie
   FAQ


Tutorials
----------
.. toctree::

   Basic Usage of TimingData and Ephemeris Objects <basic_usage>
   Special TimingData Object Usage <special_usage>
   Generating an Observing Schedule <observing_schedule>


Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


The developers were supported by grants from NASA's TESS Cycle-5
Guest Investigator program (NNH21ZDA001N-TESS), Exoplanets Research
Program (NNH21ZDA001N-XRP), and from the Idaho Space Grant
Consortium.