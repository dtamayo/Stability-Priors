Stability Constrained Characterization of Multiplanet Systems
*************************************************************

As a way to help future stability analyses, we provide the scripts used to generate the figures in Tamayo, Gilbertson and Foreman-Mackey (2020).

Reading the paper will save you some time! We recommend

* Weighting configurations by their respective SPOCK probabilities (see figure scripts, and Secs 2.4 and 4.1)
* Removing crossing configurations a priori (Sec 4.3, and see GenerateNbodyAndSPOCKPredictions.ipynb for a simple way to do that)

A good starting point to modify for future analyses is GenerateNbodyAndSPOCKPredictions.ipynb, which generates the stability csv used to generate figures.

The csvs required to reproduce Figs 3-6 are included in the repo, but the ones for the transit durations and TTV comparisons in Figs 1 and 2 are too large
You can regenerate them by running GenerateTTVandTDcsvs.ipynb
You will have to download the accompanying data to `Hadden and Lithwick 2017 <https://iopscience.iop.org/article/10.3847/1538-3881/aa71ef/meta>`_.

Figure scripts are in figures/
If you would like to regenerate our initial conditions, look at the README.txt in runNbody
