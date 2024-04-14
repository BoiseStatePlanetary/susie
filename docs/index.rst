.. Susie documentation master file, created by
   sphinx-quickstart on Sun Oct 15 12:14:46 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Susie's documentation!
=================================

Susie is a package built for astronomers to estimate rates of orbital decay due to tidal interactions in their 
extrasolar system. 

.. figure:: /images/sp_susie.png

About
-----

Susie works by fitting your transit mid-times (or occultation mid-times if given) to both a linear and quadratic model, 
then calculating a modified :math:`\chi ^2` metric called :math:`\rm BIC` for both models to determine which model 
best represents your data. Whichever model has a larger value of :math:`BIC` will be the model that best represents 
your data. If a linear model provides a better fit, your system is assumed to not be exhibiting tidal decay. 
If a quadratic model provides a better fit, your system is assumed to be exhibiting tidal decay. Metrics are simplified 
for you by calculating a :math:`\Delta BIC` value (the linear :math:`BIC` minus the quadratic :math:`BIC`). The higher
your :math:`\Delta BIC` value, the more likely your system is exhbiting tidal decay. More data on these metrics and how
they are calculated can be found in `our team's paper here <https://arxiv.org/abs/2308.04587>`_. 

You can choose to fit your data to a specified model, or we can do the work for you, fitting both models with your data 
and returning a :math:`\Delta BIC` value for you to further assess. Visualizations are given for you to further examine 
your results with your resturned model data. Just input your data into the :ref:`transit_times.py <transit_times_label>`
object, then insert your created object into the :ref:`ephemeris.py <ephemeris_label>` object to proceed with your choice
of modeling and visualizations. See the documentation and example scripts below for more.

Future work includes implementation of `Astroplan <https://astroplan.readthedocs.io/en/stable/index.html>`_ to provide
future observing schedules so you can make sure to catch every transit available from your observing point. We also
plan to improve our model fits with the `Emcee <https://emcee.readthedocs.io/en/stable/>`_ package, as well as include
functionality to detect precession that may masquerade as tidal decay in your system.

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

      from susie import Ephemeris, TransitTimes


Documentation
-------------
.. toctree::

   susie
   FAQ


Tutorials
----------
.. toctree::

   Installation <tutorial1>
   Adding Transit Data <tutorial2>
   Creating Ephemeris Models and Data <tutorial3> 
   Creating Ephemeris Visualizations <tutorial4> 


Indices and tables
------------------
* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


The developers were supported by a TESS Guest Investigator Cycle-5 grant (NNH21ZDA001N), and 
an Exoplanets Research grant (NNH21ZDA001N).