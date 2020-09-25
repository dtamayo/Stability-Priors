Stability Constrained Characterization of Multiplanet Systems
*************************************************************

As a way to help future stability analyses, we provide the scripts used to generate the figures in Tamayo, Gilbertson and Foreman-Mackey (2020): `http://arxiv.org/abs/2009.11831 <http://arxiv.org/abs/2009.11831>`_. See the figures/ folder.

Reading the paper will save you some time! We recommend:

* Weighting configurations by their respective SPOCK probabilities (see figure scripts, and Secs 2.4 and 4.1)
* Removing crossing configurations a priori (Sec 4.3, and see GenerateNbodyAndSPOCKPredictions.ipynb for a simple way to do that)

A good starting point to modify for future analyses is GenerateNbodyAndSPOCKPredictions.ipynb, which generates the stability csv used to generate figures.
To run it you will need the simulation archives for all our initial conditions, which you can get on `zenodo <https://zenodo.org/record/4048696#.X20PrC2ZPVs>`_.
Place the .tar.gz file at the root level of the repository (where the csvs, data etc. folders are), and::

    tar -xzvf K23uniform.tar.gz

which will extract it where the scripts expect them to be.

The csvs required to reproduce Figs 3-6 are included in the repo, but the ones for the transit durations and TTV comparisons in Figs 1 and 2 are too large. 
You can regenerate them by running GenerateTTVandTDcsvs.ipynb. 
You will have to download the accompanying data to `Hadden and Lithwick 2017 <https://iopscience.iop.org/article/10.3847/1538-3881/aa71ef/meta>`_.

If you would like to regenerate our initial conditions, look at the README.txt in runNbody.


