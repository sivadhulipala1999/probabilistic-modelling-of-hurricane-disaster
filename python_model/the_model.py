import pymc3 as pm
import numpy as np
import pandas as pd
import arviz as az
import theano.tensor as tt
from scipy.stats import norm
from scipy.special import logit
import time

"""
TODO: fix synthetic data
TODO: check if the parameters match for synthetic, observed data 
"""

# these parameters are being defined based on the initial setting done in author's code
theta1_theta2_mu_dict = {"bad": [220, 0.15], "medium": [270, 0.15], "good": [320, 0.15]}
theta1_theta2_sd_dict = {"bad": [25, 0.03], "medium": [25, 0.03], "good": [25, 0.03]}

# chosen based on the observations made by the authors. Represents the maximum wind speed observed across multiple areas. Can be made so that it is different for different building types
norm_factor = 300


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

theta_mus = theta1_theta2_mu_dict[BUILDING_TYPE]
theta_sds = theta1_theta2_sd_dict[BUILDING_TYPE]
theta1_mu = theta_mus[0]/norm_factor
theta2_mu = theta_mus[1]
theta1_sd = theta_sds[0]/norm_factor
theta2_sd = theta_sds[1]

# synthetic data prep
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


print(f"theta 3: mu - {theta3_mu}, sd - {theta3_sd}")
print(f"theta 4: mu - {theta4_mu}, sd - {theta4_sd}")
print(f"theta 5: mu - {theta5_mu}, sd - {theta5_sd}")
print(f"theta 6: mu - {theta6_mu}, sd - {theta6_sd}")

# Define the wind speed values for simulation
epsilon = 0.001

lower_v = 180
upper_v = 290

size = 250
v = []

for _ in range(size):
    v.append(np.random.randint(lower_v, upper_v))

v = np.array(v)/norm_factor

pi0 = inv_logit(theta3_mu + theta4_mu * np.mean(v))
pi1 = inv_logit(theta5_mu + theta6_mu * np.mean(v))

# Simulate the damage ratio data
mu_obs = phi(np.log(np.mean(v/theta1_mu)) * (1/theta2_mu))

y_zero_obs = [np.random.choice([0, 1], p=[pi0, 1 - pi0]) for _ in range(size)]
y_one_obs = [np.random.choice([0, 1], p=[1 - (1 - pi0) * pi1, (1 - pi0) * pi1]) for _ in range(size)]
y_beta_obs = np.random.beta(mu_obs + epsilon, (1-mu_obs) + epsilon, size=size)

st = time.time()

with pm.Model():
    # Priors for unknown model parameters

    theta1 = pm.Normal("theta1", mu=theta1_mu, sigma=theta1_sd)
    theta2 = pm.Normal("theta2", mu=theta2_mu, sigma=theta2_sd)
    theta3 = pm.Normal("theta3", mu=theta3_mu, sigma=theta3_sd)
    theta4 = pm.Normal("theta4", mu=theta4_mu, sigma=theta4_sd)
    theta5 = pm.Normal("theta5", mu=theta5_mu, sigma=theta5_sd)
    theta6 = pm.Normal("theta6", mu=theta6_mu, sigma=theta6_sd)
    precision = pm.Uniform("precision", 0, 100) # based on the original code

    # Expected value of outcome
    PI_0 = pm.Deterministic("PI_0", pm.math.invlogit(theta3 + theta4 * v))
    PI_1 = pm.Deterministic("PI_1", pm.math.invlogit(theta5 + theta6 * v))
    # mu = 0.5 * (1 + tt.erf((tt.log(v/theta1)/theta2 - 0)/np.sqrt(2)))
    mu = pm.Deterministic("mu", pm.math.invprobit((1/theta2_mu) * pm.math.log(v/theta1_mu)))

    # Likelihood (sampling distribution) of observations
    y_zero = pm.Bernoulli("y_zero", p=PI_0,  observed=y_zero_obs)
    y_one = pm.Bernoulli("y_one", p=PI_1, observed=y_one_obs)
    y_beta = pm.Beta("y_beta", alpha=mu * precision, beta=(1 - mu) * precision, observed=y_beta_obs)

    trace = pm.sample(1500, tune=1000, chains=3, cores=1) # tuned samples are being discarded

et = time.time()

print(f"Total time elapsed for NUTS sampling: {et - st}")

az.plot_trace(data=trace)
az.summary(trace)
