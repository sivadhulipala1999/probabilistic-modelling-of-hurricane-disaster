import pymc3 as pm
import numpy as np
import pandas as pd
import arviz as az
import theano.tensor as tt
from scipy.stats import norm
from scipy.special import logit
import time

bad_data = pd.read_csv("observations_bad.csv")
bad_data = bad_data.to_numpy()

# these parameters are being defined based on the initial setting done in author's code
theta1_theta2_mu_dict = {"bad": [220, 0.15], "medium": [270, 0.15], "good": [320, 0.15]}
theta1_theta2_sd_dict = {"bad": [25, 0.03], "medium": [25, 0.03], "good": [25, 0.03]}

# chosen based on the observations made by the authors. Represents the maximum wind speed observed across multiple areas. Can be made so that it is different for different building types
# norm_factor = 296
norm_factor = max(bad_data[:, 0])
y_observed = bad_data[:, 1]


def inv_logit(x):
    return np.exp(x)/(1 + np.exp(x))


def phi(x):
    return norm.cdf(x)


def get_mu_sd(lower_mu, prob_lower_mu, upper_mu, prob_upper_mu, building_type):
    theta1_mu, theta2_mu = theta1_theta2_mu_dict[building_type]

    wind_speed_lower_mu = np.exp(norm.ppf(lower_mu) * theta2_mu + np.log(theta1_mu/norm_factor))
    logit_lower_mu = logit(prob_lower_mu)

    wind_speed_upper_mu = np.exp(norm.ppf(upper_mu) * theta2_mu + np.log(theta1_mu/norm_factor))
    logit_upper_mu = logit(prob_upper_mu)

    coeffs = np.array([[1, wind_speed_lower_mu], [1, wind_speed_upper_mu]])
    mus = np.array([logit_lower_mu, logit_upper_mu])
    thetas = np.linalg.solve(coeffs, mus)
    return thetas[0], thetas[1]


# We want theta3, theta4 to be defined in such a way that probability pi_0 is 0.99 when mu_y is 0.01 and 0.01 when the latter 0.05
# For theta5, theta6, pi_1 is expected to be 0.01 when mu_y is 0.95 and 0.99 when mu_y is 0.99
# sd is calculated to be 10% of the mean

BUILDING_TYPE = "good"

pi_0_params = get_mu_sd(lower_mu=0.01, prob_lower_mu=0.99, upper_mu=0.05, prob_upper_mu=0.01, building_type=BUILDING_TYPE)
pi_1_params = get_mu_sd(lower_mu=0.95, prob_lower_mu=0.01, upper_mu=0.99, prob_upper_mu=0.99, building_type=BUILDING_TYPE)

# True parameter values
theta3_mu = pi_0_params[0]
theta3_sd = abs(pi_0_params[0]/10)

theta4_mu = pi_0_params[1]
theta4_sd = abs(pi_0_params[1]/10)

theta5_mu = pi_1_params[0]
theta5_sd = abs(pi_1_params[0]/10)

theta6_mu = pi_1_params[1]
theta6_sd = abs(pi_1_params[1]/10)

print(f"theta3 - mu: {theta3_mu}, sd: {theta3_sd}")
print(f"theta4 - mu: {theta4_mu}, sd: {theta4_sd}")
print(f"theta5 - mu: {theta5_mu}, sd: {theta5_sd}")
print(f"theta6 - mu: {theta6_mu}, sd: {theta6_sd}")

st = time.time()

with pm.Model():
    # Priors for unknown model parameters
    theta_mus = theta1_theta2_mu_dict[BUILDING_TYPE]
    theta_sds = theta1_theta2_sd_dict[BUILDING_TYPE]
    theta1_mu = theta_mus[0]
    theta2_mu = theta_mus[1]
    theta1_sd = theta_sds[0]
    theta2_sd = theta_sds[1]

    theta1 = pm.Normal("theta1", mu=theta1_mu, sigma=theta1_sd)
    theta2 = pm.Normal("theta2", mu=theta1_mu, sigma=theta1_sd)
    theta3 = pm.Normal("theta3", mu=theta3_mu, sigma=theta3_sd)
    theta4 = pm.Normal("theta4", mu=theta4_mu, sigma=theta4_sd)
    theta5 = pm.Normal("theta5", mu=theta5_mu, sigma=theta5_sd)
    theta6 = pm.Normal("theta6", mu=theta6_mu, sigma=theta6_sd)
    precision = pm.Uniform("precision", 0, 100) # based on the original code

    v = pm.Uniform("wind_speeds", 180, 290) # assume this distribution based on the data available in the observation files

    # Expected value of outcome
    PI_0 = pm.Deterministic("PI_0", pm.math.invlogit(theta3 + theta4 * v))
    PI_1 = pm.Deterministic("PI_1", pm.math.invlogit(theta5 + theta6 * v))
    mu = 0.5 * (1 + tt.erf((tt.log(v/theta1)/theta2 - 0)/np.sqrt(2)))

    epsilon = 0.000001

    # Likelihood (sampling distribution) of observations
    y_zero = pm.Uniform.dist( 0, epsilon)
    y_one = pm.Uniform.dist(1-epsilon, 1)
    y_beta = pm.Beta.dist(alpha=precision * mu, beta=(1 - mu) * precision)

    weights = [PI_0, (1 - PI_0) * PI_1, (1 - PI_0) * (1 - PI_1)]
    dists = [y_zero, y_one, y_beta]

    mixture = pm.Mixture("mixture", w=weights, comp_dists=dists)

    y_like = pm.Bernoulli("y_like", p=mixture, observed=y_observed)

    trace = pm.sample(10000, tune=1000, chains=3, cores=1) # tuned samples are being discarded

et = time.time()

print(f"Total time elapsed for NUTS sampling: {et - st}")

az.plot_trace(data=trace)
az.summary(trace)
