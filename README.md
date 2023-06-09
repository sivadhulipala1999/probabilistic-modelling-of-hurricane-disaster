## Abstract
Rapid impact assessments immediately after disasters are crucial to enable rapid and effective mobilization of resources for response and recovery efforts. These assessments are often performed by analysing the three components of risk: hazard, exposure and vulnerability. Vulnerability curves are often constructed using historic insurance data or expert judgments, reducing their applicability for the characteristics of the specific hazard and building stock. Therefore, this paper outlines an approach to the creation of event-specific vulnerability curves, using Bayesian statistics (i.e., the zero-one inflated beta distribution) to update a pre-existing vulnerability curve (i.e., the prior) with observed impact data derived from social media. The approach is applied in a case study of Hurricane Dorian, which hit the Bahamas in September 2019. We analysed footage shot predominantly from unmanned aerial vehicles (UAVs) and other airborne vehicles posted on YouTube in the first 10 days after the disaster. Due to its Bayesian nature, the approach can be used regardless of the amount of data available as it balances the contribution of the prior and the observations.

## Setup
1. Install Python 3.6+ with modules `pandas`, `gdal`, `osgeo`, `pingouin` and `statsmodels`.
2. Install R with packages `rjags`, `ggmcmc`, `reshape2` and `gtools`.

## Running
1. Run `parse_observations.py` using Python. This script parses the Excel tables with damage ratios and extracts the corresponding maximum wind speed from `max_wind_field.tif`. This results in 3 files: `observations_bad.csv`, `observations_medium.csv` and `observations_good.csv` with the `x`-variable as wind speed and `y`-variable as the damage ratio.
2. Run `estimate_posterior.R` using R. This takes as input the three previously generated observations and outputs samples of the posterior distribution `posteriors_bad.csv`, `posteriors_medium.csv` and `posteriors_good.csv`.

## Clone of the actual repository for our own reference 