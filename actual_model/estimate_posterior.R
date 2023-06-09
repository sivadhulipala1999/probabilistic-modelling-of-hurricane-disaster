library(rjags)
library(ggmcmc)
library(reshape2)
library(gtools)


updating <- function (name, priors_beta_mu1, priors_beta_sd1, priors_beta_mu0, priors_beta_sd0) {
  data <- read.csv(paste0(getwd(), '/observations_', name, '.csv'), header = TRUE)
  data$x < data$x
  n <- nrow(data)
  
  norm_factor <- max(data$x)
  data$x_norm <- data$x / norm_factor
  x <- data$x_norm
  y <- data$y
  
  x_plot <- seq(5, 500, 1) # seq is similar np.arange()
  x_plot_norm <- x_plot / norm_factor

  b1.prior_mu <- priors_beta_mu1 / norm_factor
  b1.prior_sd <- priors_beta_sd1 / norm_factor

  # At this point, we have normalized the data's x, y values and have also scaled down the mu and sd of the prior distribution of alpha to suit the needs

  calc_x <- function(x) {
  # qnorm - determines the boundary value given the area. qnorm(0.85) gives the 85th percentile of a normal distribution. This is the integral of exp((-x^2)/2) from 0 till the input argument
    exp((qnorm(x) - (-1 / priors_beta_mu0) * log(priors_beta_mu1 / norm_factor)) * priors_beta_mu0)
  }
  # this function is used to calculate the value of the wind velocity based on equation 5 -> exp(phi_inv(mu_y) * priors_beta_mu1) * priors_beta_mu0  = v
  # again priors_beta_mu1 is the alpha and priors_beta_mu0 is the beta of the equation corresponding to the vulnerability curve

  get_coeficients <- function(lower, lower_bern, upper, upper_bern) {
    points.x <- c(calc_x(lower), calc_x(upper)) # c() creates a vector with the given values
    points.y <- c(logit(lower_bern), logit(upper_bern))
    priors.alpha <- solve(cbind(1, points.x), points.y) # cbind is used to merge 2 dataframe together
    # alphas are related to priors 0, 1 - their parameters - theta3, theta4, theta5, theta6
    a <- list("a0"=as.numeric(priors.alpha[1]), "a1"=as.numeric(priors.alpha[2]))
    return(a)
  }
  # what coefficients are we getting here? - we are getting a0, a1 which are used to find the prior distributions for 0 and 1

  get_a_mu_sd <- function(
    lower_m,
    lower_m_bern,
    upper_m,
    upper_m_bern) {
    a_m <- get_coeficients(lower_m, lower_m_bern, upper_m, upper_m_bern)
    list(
      "mu0"=a_m$a0,
      "sd0"=abs(a_m$a0 / 10),
      "mu1"=a_m$a1,
      "sd1"=abs(a_m$a1 / 10) # does not want to deviate from the mu from the solution a lot, explained in appendix D
    )
  }

  # appendix D explains that they chose these values to mimic Pi 0, Pi 1's expected distributions
  # lower_m, upper_m both give the lower and upper values of mu_y and the lower_m_bern, upper_m_bern explain the corresponding values of the probability distributions (Pi)
  priors.zero <- get_a_mu_sd(0.01, 0.99, 0.05, 0.01)
  priors.one <- get_a_mu_sd(0.95, 0.01, 0.99, 0.99)

  iterations <- 1000
  priors.sample <- data.frame(
    rep(x_plot, times=iterations),
    rep(x_plot_norm, times=iterations),
    rep(1:iterations, each=length(x_plot))
  )
  colnames(priors.sample) <- c("x_plot", "x_plot_norm", "iteration")

  # draw a 1000 samples from the normal distributions for all thetas based on the mean and standard deviation values each of them have
  priors.zero.random0 <- rnorm(iterations, priors.zero$mu0, priors.zero$sd0) # theta 3 distribution
  priors.zero.random1 <- rnorm(iterations, priors.zero$mu1, priors.zero$sd1) # theta 4 distribution
  priors.one.random0 <- rnorm(iterations, priors.one$mu0, priors.one$sd0) # theta 5 distribution
  priors.one.random1 <- rnorm(iterations, priors.one$mu1, priors.one$sd1) # theta 6 distribution

  priors.mean <- data.frame(
    x_plot, x_plot_norm
  )

  bernoulli_f <- function(a, b, x) {
    inv.logit(a + b * x)
  }

  # calculating the mean of the prior distributions of pi_0, pi_1 based on the means that we just calculated before
  priors.mean$zero <- bernoulli_f(priors.zero$mu0, priors.zero$mu1, priors.mean$x_plot_norm)
  priors.mean$one <- bernoulli_f(priors.one$mu0, priors.one$mu1, priors.mean$x_plot_norm)

  bernoulli_f_i <- function(a, b, i, x) {
    bernoulli_f(a[i], b[i], x)
  }

  # bernoulli_f_i is being used for the probabilities of 0 and 1

  priors.sample$zero <- bernoulli_f_i(priors.zero.random0, priors.zero.random1, priors.sample$iteration, priors.sample$x_plot_norm) # pi 0
  priors.sample$one <- bernoulli_f_i(priors.one.random0, priors.one.random1, priors.sample$iteration, priors.sample$x_plot_norm) # pi 1

  # draw a thousand samples for the beta distribution
  priors.beta.random0 <- rnorm(iterations, priors_beta_mu0, priors_beta_sd0)
  priors.beta.random1 <- rnorm(iterations, priors_beta_mu1, priors_beta_sd1)

  beta_f <- function(a, b, x) {
    pnorm((1 / a) * log((x * norm_factor) / b)) # pnorm calculates the CDF where x is normal. So this is cumulative lognormal distribution which corresponds to the y equation (eq 2)
  }

  priors.mean$beta <- beta_f(priors_beta_mu0, priors_beta_mu1, priors.mean$x_plot_norm)

  beta_f_i <- function(a, b, i, x) {
    beta_f(a[i], b[i], x)
  }

  priors.sample$beta <- beta_f_i(priors.beta.random0, priors.beta.random1, priors.sample$iteration, priors.sample$x_plot_norm)

  write.csv(priors.sample, paste(getwd(), '/data/vulnerability/priors_', name, '.csv', sep=""))

  # split the data into discrete and continuous components
  y.zero <- ifelse(y == 0, y, NA)
  y.one <- ifelse(y == 1, y, NA)
  y.isone <- ifelse(is.na(y.one), 0, 1)
  y.iszero <- ifelse(is.na(y.zero), 0, 1)

  y.zero <- y.zero[!is.na(y.zero)]
  x.zero <- x[y.iszero == 1]
  y.one <- y.one[!is.na(y.one)]
  x.one <- x[y.isone == 1]

  n.zero <- length(y.zero)
  n.one <- length(y.one)

  which.cont <- which(y < 1 & y > 0)
  y.c <- ifelse(y < 1 & y > 0, y, NA)
  y.c <- y.c[!is.na(y.c)]
  n.cont <- length(y.c)
  x.c <- x[which.cont]

  # write model
  cat(
    "
    model{
      # priors
      # dnorm is specified in terms of mean and precision. It gives the entire pdf of the normal distribution (discrete)
      zero0 ~ dnorm(priors.zero_mu0, 1 / (priors.zero_sd0 ^ 2))
      zero1 ~ dnorm(priors.zero_mu1, 1 / (priors.zero_sd1 ^ 2))
      one0 ~ dnorm(priors.one_mu0, 1 / (priors.one_sd0 ^ 2))
      one1 ~ dnorm(priors.one_mu1, 1 / (priors.one_sd1 ^ 2))
      b0 ~ dnorm(priors.b0_mu, 1 / (priors.b0_sigma ^ 2))  # sigma to tau (tau = 1 / sigma**2) - sigma here represents precision
      b1 ~ dnorm(priors.b1_mu, 1 / (priors.b1_sigma ^ 2))  # sigma to tau (tau = 1 / sigma**2)
      phi ~ dunif(0, 100)
      # The default link function is log in betareg
      # https://rdrr.io/cran/betareg/man/betareg.html
      # log(phi) <- t0

      # eqs 6, 7
      for (i in 1:n) {
        logit(alpha_zero[i]) <- zero0 + zero1 * x[i]
        y.iszero[i] ~ dbern(alpha_zero[i]) # pi 0
        logit(alpha_one[i]) <- one0 + one1 * x[i]
        y.isone[i] ~ dbern(alpha_one[i]) # pi 1
      }

      # do not know
      for (i in 1:n.zero) {
        y.zero[i] ~ dbern(0) # the probability of success p here is estimated in BUGS, based on the value of y.zero
      }
      for (i in 1:n.one) {
        y.one[i] ~ dbern(1) # the probability of success here is estimated based on the value of y.one
      }

      # likelihood for mu and phi (precision) - probit link function
      for (i in 1:n.cont) {
        y.c[i] ~ dbeta(p[i], q[i]) # y.c = cont values - eq 3 -> Here we estimate the parameters p, q based on y.c
        p[i] <- mu2[i] * phi # based on the value of p estimated, mu2, phi are estimated here
        q[i] <- (1 - mu2[i]) * phi # same with the above thing
        probit(mu2[i]) <- (1 / b0) * log(x.c[i] / b1) # eq 5
      }
    }
    ", file= "beinf.txt"
  )

  jd <- list(
    priors.zero_mu0=priors.zero$mu0,
    priors.zero_sd0=priors.zero$sd0,
    priors.zero_mu1=priors.zero$mu1,
    priors.zero_sd1=priors.zero$sd1,
    priors.one_mu0=priors.one$mu0,
    priors.one_sd0=priors.one$sd0,
    priors.one_mu1=priors.one$mu1,
    priors.one_sd1=priors.one$sd1,
    priors.b0_mu=priors_beta_mu0,
    priors.b0_sigma=priors_beta_sd0,
    priors.b1_mu=b1.prior_mu,
    priors.b1_sigma=b1.prior_sd,
    x=x,
    y.zero=y.zero,
    y.one=y.one,
    y.c=y.c,
    y.iszero = y.iszero,
    y.isone = y.isone,
    n.zero=n.zero,
    n.one=n.one,
    n.cont = n.cont,
    x.c=x.c,
    n=n
  )
  model <- jags.model("beinf.txt", data=jd, n.chains=3, n.adapt=1000) # create the model such that it runs a 1000 times before proceeding, a.k.a burn in
  # n.chains = parallel mcmc chains argument

  update(model, 1000)

  out <- coda.samples(model, c("zero0", "zero1", "one0", "one1", "b0", "b1", "phi"),
                      n.iter=100000, thin = 100)
  # this is for generating actual samples from the posterior distribution based on the model

  ggd <- ggs(out)

  phi.post <- subset(ggd, Parameter == "phi")$value

  zero0.post <- subset(ggd, Parameter == "zero0")$value
  zero1.post <- subset(ggd, Parameter == "zero1")$value
  one0.post <- subset(ggd, Parameter == "one0")$value
  one1.post <- subset(ggd, Parameter == "one1")$value
  b0.post <- subset(ggd, Parameter == "b0")$value
  b1.post <- subset(ggd, Parameter == "b1")$value
  phi.post <- subset(ggd, Parameter == "phi")$value

  thetas <- data.frame(zero0.post, zero1.post, one0.post, one1.post, b0.post, b1.post, phi.post)
  colnames(thetas) <- c("theta_3", "theta_4", "theta_5", "theta_6", "theta_2", "theta_1", "phi")
  write.csv(thetas, paste(getwd(), '/data/vulnerability/posterior_thetas_', name, '.csv', sep=""))

  n.stored <- length(phi.post)

  P.iszero <- array(dim=c(n.stored, length(x_plot)))
  for (i in 1:length(x_plot)){
    P.iszero[, i] <- inv.logit(zero0.post + zero1.post * x_plot_norm[i])
  }
  pdzero <- melt(P.iszero, varnames = c("iteration", "site"),
                 value.name = "Pr.iszero")
  pdzero$x <- x_plot[pdzero$site]

  P.isone <- array(dim=c(n.stored, length(x_plot)))
  for (i in 1:length(x_plot)){
    P.isone[, i] <- inv.logit(one0.post + one1.post * x_plot_norm[i])
  }
  pdone <- melt(P.isone, varnames = c("iteration", "site"),
                value.name = "Pr.isone")
  pdone$x <- x_plot[pdone$site]

  expect <- array(dim=c(n.stored, length(x_plot)))
  for (i in 1:length(x_plot)){
    expect[, i] <- pnorm((1 / b0.post) * log(x_plot_norm[i] / b1.post))
  }
  exd <- melt(expect, varnames=c("iteration", "site"), value.name = "Expectation")
  exd$x <- x_plot[exd$site]

  posteriors.sample <- data.frame(exd$x, exd$iteration, pdzero$Pr.iszero, pdone$Pr.isone, exd$Expectation)
  colnames(posteriors.sample) <- c("x_plot", "iteration", "zero", "one", "beta")
  write.csv(posteriors.sample, paste0(getwd(), '/data/vulnerability/posteriors_', name, '.csv'))

  file.remove('beinf.txt')
}

updating('bad', 220, 25, 0.15, 0.03)
updating('medium', 270, 25, 0.15, 0.03)
updating('good', 320, 25, 0.15, 0.03)

# priors_beta_mu1, priors_beta_mu0 -> Represent the values of alpha, beta respectively for the y equation (2). This is the prior distribution of damage ratio against the wind speed
# refer to figure 4 to see the curves and the corresponding values. The standard deviation values were chosen as a educational guess most likely