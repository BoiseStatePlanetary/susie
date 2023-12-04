---
title: 'Susie: Tools for Detecting Tidal Decay'
tags:
  - Python
  - astronomy
  - exoplanets
  - tidal decay
authors:
  - name: Malia Barker
    affiliation: 1
  - name: Holly VanLooy
    # equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Rachel Huchmala
    affiliation: 1
  - name: Brian Jackson
    affiliation: 1
  - name: Elisabeth Adams
    affiliation: 2
affiliations:
 - name: Boise State University, USA
   index: 1
 - name: Planetary Science Institute, USA
   index: 2
date: 05 October 2023
bibliography: paper.bib

# TODO: Maybe delete?
# # Optional fields if submitting to a AAS journal too, see this blog post:
# # https://blog.joss.theoj.org/2018/12/a-new-collaboration-with-aas-publishing
# aas-doi: 10.3847/xxxxx <- update this with the DOI from AAS once you know it.
# aas-journal: Astrophysical Journal <- The name of the AAS journal.
---

# Summary
<!-- A summary describing the high-level functionality and purpose of the software for a diverse, non-specialist audience. -->
The goal of this package is to take transit-timing observational data and give users access to different methods and visualization to learn more about the possible tidal decay of the exoplanet. The package consists of two main objects: transit time and ephemeris. Users insert observational data into the transit time object. 

Transit time data consists of a string denoting the time format and time scale of the data and Numpy lists of epochs, mid-transit times, and uncertainties for the mid-transit times. The object will always default to using the Barycentric Julian Date (BJD) time format with the Barycentric Dynamical Time (TDB) scale. (This following is a maybe; I still need to talk to Elisabeth) If the user indicates their times are not corrected for barycentric light travel times, the Astropy time, coordinates, and unit packages will be used for corrections. 

The ephemeris object takes in the user’s created transit time object as data. The primary method of the object fits the user’s timing transit data to a specified model ephemeris, either linear or quadratic. The strategy and factory python design patterns are used to create the model ephemeris. The strategy pattern allows us to dynamically choose the model fit method based on conditionals in runtime. We define an abstract class to represent the model fit method. The linear and quadratic model fit methods are built on the abstract class and contain the respective equations for the curve fit package. The factory pattern allows us to create objects based on conditionals during runtime. The user inputs their desired ephemeris type (linear or quadratic), which is used as a conditional passed to the factory pattern to decide which model fit will be executed. 

The user is returned a Python dictionary consisting of the model parameters and the calculated values of the model at each epoch of their inputted transit timing object. This data can then be used for further calculations and visualizations that tell the user about the motion of the exoplanet. 

Python tests utilize pytest and pytest mock objects to simulate data and object creation. Tests ensure objects only accept user inputs of the correct types, method outputs are of the correct type and length, and calculation methods output correct numerical values. 

Our package is publicly available to download through the Python Package Index (PyPi). The package uses the setuptools python package as the build system, which uses inputs from the metadata file to automate the build process. To upload distribution archives created using our build system, we use the official PyPi upload tool twine. Users can download the package from the PyPi repository using the Python package installer, pip, in the command line or a Jupyter notebook. Additionally, the code is publicly accessible on GitHub.

# Statement of need
<!-- A clear statement of need that illustrates the purpose of the software. -->

# Similar Tools
<!-- A description of how this software compares to other commonly-used packages in this research area. -->

# Mathematics
Single dollars ($) are required for inline mathematics e.g. $f(x) = e^{\pi/x}$

Double dollars make self-standing equations:

$$\Theta(x) = \left\{\begin{array}{l}
0\textrm{ if } x < 0\cr
1\textrm{ else}
\end{array}\right.$$

You can also use plain \LaTeX for equations
\begin{equation}\label{eq:fourier}
\hat f(\omega) = \int_{-\infty}^{\infty} f(x) e^{i\omega x} dx
\end{equation}
and refer to \autoref{eq:fourier} from text.

# Citations
Citations to entries in paper.bib should be in
[rMarkdown](http://rmarkdown.rstudio.com/authoring_bibliographies_and_citations.html)
format.

If you want to cite a software repository URL (e.g. something on GitHub without a preferred
citation) then you can do it with the example BibTeX entry below for @fidgit.

For a quick reference, the following citation commands can be used:
- `@author:2001`  ->  "Author et al. (2001)"
- `[@author:2001]` -> "(Author et al., 2001)"
- `[@author1:2001; @author2:2001]` -> "(Author1 et al., 2001; Author2 et al., 2002)"

# Figures
Figures can be included like this:
![Caption for example figure.\label{fig:example}](figure.png)
and referenced from text using \autoref{fig:example}.

Figure sizes can be customized by adding an optional second parameter:
![Caption for example figure.](figure.png){ width=20% }

# Acknowledgements
<!-- Mentions (if applicable) of any ongoing research projects using the software or recent scholarly publications enabled by it. NOTE: Don't know if this would be put here. -->

# References
<!-- A list of key references including a link to the software archive. -->