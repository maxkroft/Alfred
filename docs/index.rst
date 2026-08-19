.. Alfred documentation master file, created by
   sphinx-quickstart on Mon Jan 12 13:51:36 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.



Alfred
======

``Alfred`` (the **A**\wesome **L**\ibrary **F**\or **R**\obust **E**\xoplanet **D**\etection) is a package designed for flexible, user-friendly 
exoplanet detection and characterization.

Key Features:

* Transit and radial velocity fitting for arbritary systems of exoplanets
* Simultaneous light curve detrending with Gaussian processes
* Optional simultaneous stellar SED fitting
* Flexible, user-defined priors
* Easy-to-understand initialization files, with tailored GUIs for each file
* Support for photometry and RVs from multiple instruments

At its core, ``Alfred`` is built around `batman <https://lkreidberg.github.io/batman/docs/html/index.html>`_
(for transit modelling), a small numerical solver for Kepler's equation (for RV modeling), `isochrones <https://isochrones.readthedocs.io/en/latest/index.html#>`_ (for stellar modelling),
and `emcee <https://emcee.readthedocs.io/en/stable/>`_ (to carry out MCMC fitting). ``Alfred`` handles all the busy-work behind the scenes,
while ``batman`` and friends do all the heavy-lifting, allowing you to focus on the science.

If you find any bugs or want to request a feature, please create an issue on the GitHub `here <https://github.com/maxkroft/Alfred/issues>`_. If you are interested in contributing, please reach out to `Max <https://maxkroft.github.io/>`_.

Attribution
+++++++++++

* If you use ``Alfred`` in your work, please cite `Kroft et al. 2026 <https://iopscience.iop.org/article/10.3847/2515-5172/ae9165>`_.
* We also recommend citing the ``emcee`` paper, `Foreman-Mackey et al. 2013 <https://ui.adsabs.harvard.edu/abs/2013PASP..125..306F/abstract>`_.
* Additionally, if you do any transit fitting, please cite ``batman`` with `Kreidberg 2015 <https://ui.adsabs.harvard.edu/abs/2015PASP..127.1161K/abstract>`_.
* Finally, if you do any stellar fitting, please cite ``isochrones`` with `Morton 2015 <https://ui.adsabs.harvard.edu/abs/2015ascl.soft03010M/abstract>`_.

User Guide
++++++++++


.. toctree::
   :maxdepth: 2

   installation
   quickstart.ipynb
   tutorials
   api


Future Features Wishlist
++++++++++++++++++++++++

* Custom prior distributions
* Alternative samplers, such as SBI
* Gaussian Process support in RV fitting
* Transit depth and duration variations
* Support for other limb darkening parameterizations
* More physically realistic TTV modeling

Changelog
+++++++++

**1.1.2 (2026-August-19)**

* Added support for parallel MCMC sampling. Users can manually set the number of cores to use, or automatically determine the optimal number by setting the parallel option to 'auto' in the fit function.

**1.1.1 (2026-July-28)**

* Better handling of empty columns in init files.
* Added default rows for empty init files.
* Add units to the InitPriorsGUI.
* Updated how gamma and RV offsets are initialized, removed subtracting mean of RV data before fitting.

**1.1.0 (2026-July-23)**

* Added GUIs to replace init file text prompts on creation and editing. Each init file type has its own tailored GUI that can be brought up by calling the init file object.

**1.0.2 (2026-July-14)**

* Corrected typo that altered a/Rs derived quantities.

**1.0.1 (2026-June-23)**

* Bug fixes.

**1.0.0 (2026-June-10)**

* Initial release.