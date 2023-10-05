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
  - name: Holly van Looey
    # equal-contrib: true # (This is how you can denote equal contributions between multiple authors)
    affiliation: 1
  - name: Brian Jackson
    affiliation: 1
  - name: Elisabeth Adams
    affiliation: 2
  - name: Rachel Huchmala
    affiliation: 1
affiliations:
 - name: Boise State University, USA
   index: 1
 - name: Planetary Science Institute, Country
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