library(xts)  # to manipulate time series of stock data
library(quantmod)  # to download stock data
library(PerformanceAnalytics)  # to compute performance measures
## https://palomar.home.ece.ust.hk/MAFS6010R_lectures/slides_risk_parity_portfolio.html#1
## MAFS6010R - Portfolio Optimization with R
## MSc in Financial Mathematics
## Hong Kong University of Science and Technology (HKUST)
## Fall 2019-20
# set begin-end date and stock namelist
begin_date <- "2013-02-01"
end_date <- "2021-11-01"
stock_namelist <- c("600079.SS","600426.SS","600499.SS","600660.SS","601899.SS")

# download data from YahooFinance
data_set <- xts()
for (stock_index in 1:length(stock_namelist))
  data_set <- cbind(data_set, Ad(getSymbols(stock_namelist[stock_index], 
                                            from = begin_date, 
                                            to = end_date, 
                                            auto.assign = FALSE)))

colnames(data_set) <- stock_namelist
indexClass(data_set) <- "Date"
str(data_set)
library(rtsdata)
ds.storage.file.csv.save(data_set, "D:\\Desktop\\Stock.csv")

plot(data_set,legend.loc = "topleft", col = rainbow10equal)
SP500_index <- Ad(getSymbols("^GSPC", from = begin_date, to = end_date, auto.assign = FALSE))
colnames(SP500_index) <- "index"
head(SP500_index)
plot(SP500_index)

prices <- data_set
# compute log-returns and linear returns
X_log <- diff(log(prices))[-1]
X_lin <- (prices/lag(prices) - 1)[-1]

# or alternatively...
X_log <- CalculateReturns(prices, "log")[-1]
X_lin <- CalculateReturns(prices)[-1]

N <- ncol(X_log)  # number of stocks
T <- nrow(X_log)  # number of days

# We can take a look at the prices of the stocks:
plot(prices/rep(prices[1, ], each = nrow(prices)), col = rainbow10equal, legend.loc = "topleft",
     main = "Normalized prices")
# We now divide the data into a training set and test set:
T_trn <- round(0.7*T)  # 70% of data
X_log_trn <- X_log[1:T_trn, ]
X_log_tst <- X_log[(T_trn+1):T, ]
X_lin_trn <- X_lin[1:T_trn, ]
X_lin_tst <- X_lin[(T_trn+1):T, ]

# sample means and sample covariance matrix
mu <- colMeans(X_log_trn)
Sigma <- cov(X_log_trn)



library(CVXR)  # interface for convex optimization solvers

# define portfolio formulations
portolioMarkowitz <- function(mu, Sigma, lmd = 0.5) {
  w <- Variable(nrow(Sigma))
  prob <- Problem(Maximize(t(mu) %*% w - lmd*quad_form(w, Sigma)),
                  constraints = list(w >= 0, sum(w) == 1))
  result <- solve(prob)
  return(as.vector(result$getValue(w)))
}

portolioGMVP <- function(Sigma) {
  w <- Variable(nrow(Sigma))
  prob <- Problem(Minimize(quad_form(w, Sigma)), 
                  constraints = list(w >= 0, sum(w) == 1))
  result <- solve(prob)
  return(as.vector(result$getValue(w)))
}

# compute portfolios
w_Markowitz <- portolioMarkowitz(mu, Sigma)
w_GMVP <- portolioGMVP(Sigma)


# We can now plot the allocation of the portfolios:
# put together all portfolios
w_all <- cbind(w_GMVP, w_Markowitz)
rownames(w_all) <- colnames(X_lin)
colnames(w_all) <- c("GMVP", "Markowitz MVP")

# plot
barplot(t(w_all), col = rainbow8equal[1:2],
        main = "Portfolio allocation", xlab = "stocks", ylab = "dollars", beside = TRUE, 
        legend = colnames(w_all))  #args.legend = list(x = "topleft", inset = 0.04)

# We can see that the Markowitz MVP concentrates all the budget in one single asset! The GMVP is much more diversified.
# Then we can compare the performance (in-sample vs out-of-sample):
# compute returns of all portfolios
ret_all <- xts(X_lin %*% w_all, index(X_lin))
ret_all_trn <- ret_all[1:T_trn, ]
ret_all_tst <- ret_all[-c(1:T_trn), ]
# performance in-sample
t(table.AnnualizedReturns(ret_all_trn))
# performance out-of-sample
t(table.AnnualizedReturns(ret_all_tst))

# We can see that the Markowitz MVP performs better than the GMVP in-sample, but much worse out-of-sample! 
# This is due to the bad estimation of ??.
{ chart.CumReturns(ret_all, main = "Performance of different portfolios", 
                   wealth.index = TRUE, legend.loc = "topleft", colorset = rich8equal)
  addEventLines(xts("training", index(X_lin[T_trn])), srt=90, pos=2, lwd = 2, col = "darkblue") }

# The cum PnL may seem contradictory at first because the Markowitz MVP seems to be 
# doing much better than the GMVP. This is however a visual effect. 
# The drawdown is very instructive:
# Let us plot the cumulative PnL over time:
{ chart.Drawdown(ret_all, main = "Drawdown of different portfolios", 
                 legend.loc = "bottomleft", colorset = rich8equal)
  addEventLines(xts("training", index(X_lin[T_trn])), srt=90, pos=2, lwd = 2, col = "darkblue") }
# We can see that the drawdown of the Markowitz mean-variance portfolio is indeed much worse than that of the GMVP.

# To study the sensitivity with respect to the parameters ?? and ??, 
# we will design multiple portfolios based on different samples from the training set:
# compute 8 different set of portfolios from different input data
w_GMVP_acc <- w_Markowitz_acc <- NULL
for (i in 1:8) {
  # sample means with random samples
  idx <- sample(1:T_trn, T_trn/2)
  mu_ <- colMeans(X_log_trn[idx, ])
  Sigma_ <- cov(X_log_trn[idx, ])
  
  # design portfolios
  w_Markowitz_acc <- cbind(w_Markowitz_acc, portolioMarkowitz(mu_, Sigma_))
  w_GMVP_acc <- cbind(w_GMVP_acc, portolioGMVP(Sigma_))
}
rownames(w_GMVP_acc) <- rownames(w_Markowitz_acc) <- colnames(X_lin)
# Let's plot different realizations of the GMVP portfolio:
barplot(t(w_GMVP_acc), col = rainbow8equal,
        main = "Different realizations of GMVP portfolio", 
        xlab = "stocks", ylab = "dollars", beside = TRUE)
# The GMVP is not very sensitive since all the realizations have a similar allocation.
# Let's plot now the different realizations of the Markowitz mean-variance portfolio:
barplot(t(w_Markowitz_acc), col = rainbow8equal,
        main = "Different realizations of Markowitz mean-variance portfolio", 
        xlab = "stocks", ylab = "dollars", beside = TRUE)

# The mean-variance portfolio is highly sensitive. 
# So sensitive that the portfolio at each realization is totally different!
# Of course the reason for this very distinct behavior is that the sensitivity 
# w.r.t. ?? is much greater than w.r.t. ?? 
# (also, the estimation error in the former is larger than in the latter) 
# (Chopra and Ziemba 1993).



# compute EWP
w_EWP <- rep(1/N, N)

# compute naive RPP
sigma2 <- diag(Sigma)
w_RPP_naive <- 1/sqrt(sigma2)
w_RPP_naive <- w_RPP_naive/sum(w_RPP_naive)

# add portfolios to the two previous ones
w_all <- cbind(w_all, 
               "EWP"         = w_EWP,
               "RPP (naive)" = w_RPP_naive)  

# plot
barplot(t(w_all), col = rainbow8equal[1:4],
        main = "Portfolio allocation", xlab = "stocks", ylab = "dollars", beside = TRUE, 
        legend = colnames(w_all))


# compute risk contributions
risk_all <- cbind("GMVP"          = as.vector(w_GMVP * (Sigma %*% w_GMVP)),
                  "Markowitz MVP" = as.vector(w_Markowitz * (Sigma %*% w_Markowitz)),
                  "EWP"           = as.vector(w_EWP * (Sigma %*% w_EWP)),
                  "RPP (naive)"    = as.vector(w_RPP_naive * (Sigma %*% w_RPP_naive)))
rownames(risk_all) <- colnames(X_lin)
RRC_all <- sweep(risk_all, MARGIN = 2, STATS = colSums(risk_all), FUN = "/")  # normalize each column

# plot
barplot(t(RRC_all), col = rainbow8equal[1:4],
        main = "Relative risk contribution", xlab = "stocks", ylab = "risk", beside = TRUE, 
        legend = colnames(RRC_all))

# Observe how the RPP has a more even risk contribution distribution than the 
# other portfolios. However, it is not perfect because the design is 
# the naive one assuming a diagonal covariance matrix.



# compute returns of all portfolios
ret_all <- xts(X_lin %*% w_all[, c("GMVP", "Markowitz MVP", "EWP", "RPP (naive)")], 
               order.by = index(X_lin))
ret_all_trn <- ret_all[1:T_trn, ]
ret_all_tst <- ret_all[-c(1:T_trn), ]

# performance in-sample
t(table.AnnualizedReturns(ret_all_trn))

# performance out-of-sample
t(table.AnnualizedReturns(ret_all_tst))

{ chart.CumReturns(ret_all, main = "Cum PnL of different portfolios", 
                   wealth.index = TRUE, legend.loc = "topleft", colorset = rainbow8equal)
  addEventLines(xts("training", index(X_lin[T_trn])), srt=90, pos=2, lwd = 2, col = "darkblue") }


{ chart.Drawdown(ret_all, main = "Drawdown of different portfolios", 
                 legend.loc = "bottomleft", colorset = rainbow6equal)
  addEventLines(xts("training", index(X_lin[T_trn])), srt=90, pos=2, lwd = 2, col = "darkblue") }





library(rootSolve)

b <- rep(1/N, N)

# function definition F(x) = Sigma %*% x - b/x
f_root <- function(x, parms) {
  Sigma <- parms
  N <- nrow(Sigma)
  return(Sigma %*% x - b/x)
}

# finding the root
x_root <- multiroot(f_root, start = b, parms = Sigma)$root
w_root <- x_root/sum(x_root)

# sanity check
Sigma %*% x_root - b/x_root




# Let's plot the risk contribution and see:
w_all <- cbind(w_all, 
               "RPP (root)" = w_root)
# compute risk contributions
risk_all <- cbind(risk_all, 
                  "RPP (root)" = as.vector(w_root * (Sigma %*% w_root)))
RRC_all <- sweep(risk_all, MARGIN = 2, STATS = colSums(risk_all), FUN = "/")  # normalize each column

# plot
barplot(t(RRC_all), col = rainbow8equal[1:5],
        main = "Relative risk contribution", xlab = "stocks", ylab = "risk", beside = TRUE, 
        legend = colnames(RRC_all))
# We can see that the RPP based on finding the root gives a perfectly 
# equalized risk contribution (better than the naive solution, of course).



# initial point
x0 <- rep(1/N, N)

# function definition
fn_convex <- function(x, Sigma) {
  N <- nrow(Sigma)
  return(0.5 * t(x) %*% Sigma %*% x - (1/N)*sum(log(x)))
}

# optimize with general-purpose solver
result <- optim(par = x0, fn = fn_convex, Sigma = Sigma, method = "BFGS")
x_convex <- result$par
w_RPP_convex <- x_convex/sum(x_convex)

# sanity check of the solution
b <- rep(1/N, N)
Sigma %*% x_convex - b/x_convex


# plot
w_all <- cbind(w_all, "RPP (convex)" = w_RPP_convex)
barplot(t(w_all), col = rainbow8equal[1:7],
        main = "Portfolio allocation", xlab = "stocks", ylab = "dollars", beside = TRUE, 
        legend = colnames(w_all))





# compute risk contributions
risk_all <- cbind(risk_all, 
                  "RPP (convex)" = as.vector(w_RPP_convex * (Sigma %*% w_RPP_convex)))
RRC_all <- sweep(risk_all, MARGIN = 2, STATS = colSums(risk_all), FUN = "/")  # normalize each column

# plot
barplot(t(RRC_all), col = rainbow8equal[1:7],
        main = "Relative risk contribution", xlab = "stocks", ylab = "risk", beside = TRUE, 
        legend = colnames(RRC_all))




# initial point
x0 <- rep(1/N, N)

# function definition
fn_nonconvex <- function(w, Sigma) {
  N <- length(w)
  risks <-  w * (Sigma %*% w)
  g <- rep(risks, times = N) - rep(risks, each = N)
  return(sum(g^2))
}

# optimize with general-purpose solver
result <- optim(par = x0, fn = fn_nonconvex, Sigma = Sigma, method = "BFGS")
x_gen_solver <- result$par
w_RPP_gen_solver <- x_gen_solver/sum(x_gen_solver)

# plot
w_all <- cbind(w_all, "RPP (gen-solver)" = w_RPP_gen_solver)
barplot(t(w_all), col = rainbow8equal[1:7],
        main = "Portfolio allocation", xlab = "stocks", ylab = "dollars", beside = TRUE, 
        legend = colnames(w_all))




# compute risk contributions
risk_all <- cbind(risk_all, 
                  "RPP (gen-solver)" = as.vector(w_RPP_gen_solver * (Sigma %*% w_RPP_gen_solver)))
RRC_all <- sweep(risk_all, MARGIN = 2, STATS = colSums(risk_all), FUN = "/")  # normalize each column

# plot
barplot(t(RRC_all), col = rainbow8equal[1:7],
        main = "Relative risk contribution", xlab = "stocks", ylab = "risk", beside = TRUE, 
        legend = colnames(RRC_all))





library(riskParityPortfolio)
#?riskParityPortfolio  # to get help for the function

# use package
rpp <- riskParityPortfolio(Sigma)
names(rpp)

# plot
w_all <- cbind(w_all, "RPP (package)" = rpp$w)
barplot(t(w_all), col = rainbow8equal[1:8],
        main = "Portfolio allocation", xlab = "stocks", ylab = "dollars", beside = TRUE, 
        legend = colnames(w_all))


# compute risk contributions
risk_all <- cbind(risk_all, 
                  "RPP (package)" = as.vector(rpp$w * (Sigma %*% rpp$w)))
RRC_all <- sweep(risk_all, MARGIN = 2, STATS = colSums(risk_all), FUN = "/")  # normalize each column

# plot
barplot(t(RRC_all), col = rainbow8equal[1:8],
        main = "Relative risk contribution", xlab = "stocks", ylab = "risk", beside = TRUE, 
        legend = colnames(RRC_all))








compute_gA <- function(w, Sigma) {
  N <- length(w)
  g <- rep(NA, N^2)
  A <- matrix(NA, N^2, N)
  for (i in 1:N) {
    Mi <- matrix(0, N, N)
    Mi[i, ] <- Sigma[i, ]
    for (j in 1:N) {
      Mj <- matrix(0, N, N)
      Mj[j, ] <- Sigma[j, ]
      #g[i + (j-1)*N]   <- t(w) %*% (Mi - Mj) %*% w
      g[i + (j-1)*N]   <- w[i]*(Sigma[i, ] %*% w) - w[j]*(Sigma[j, ] %*% w)  # faster
      A[i + (j-1)*N, ] <- (Mi + t(Mi) - Mj - t(Mj)) %*% w
    }
  }
  # # g can be computed much more efficiently with this code:
  # wSw <- w * (Sigma %*% w)
  # g <- rep(wSw, times = N) - rep(wSw, each = N)
  return(list(g = g, A = A))
}




# Now we can implement the main loop of the SCA algorithm:
library(quadprog)  # install.packages("quadprog")

# parameters
max_iter <- 40
tau <- 1e-6
zeta <- 0.1
gamma <- 0.99
# initial point
obj_value <- NULL
w_SCA <- rep(1/N, N)  # initial point
for (k in 1:max_iter) {
  # compute parameters for QP
  gA <- compute_gA(w_SCA, Sigma)
  g <- gA$g
  A <- gA$A
  Q <- 2 * t(A) %*% A + tau*diag(N)  # faster code is: crossprod(A) = t(A) %*% A
  q <- 2 * t(A) %*% g - Q %*% w_SCA
  obj_value <- c(obj_value, sum(g^2))
  
  # # solve the inner QP with CVXR
  # w_ <- Variable(N)
  # prob <- Problem(Minimize(0.5*quad_form(w_, Q) + t(q) %*% w_),
  #                 constraints = list(sum(w_) == 1))
  # result <- solve(prob)
  # w_ <- as.vector(result$getValue(w_))
  
  # solve the problem with solve.QP() which is much faster than CVXR)
  w_ <- solve.QP(Q, -q, matrix(1, N, 1), 1, meq = 1)$solution
  
  # next w
  gamma <- gamma*(1 - zeta*gamma)
  w_SCA_prev <- w_SCA
  w_SCA <- w_SCA + gamma*(w_ - w_SCA)
  
  # stopping criterion
  if (max(abs(w_SCA - w_SCA_prev)) <= 1e-4*max(abs(w_SCA_prev)))
    break
  # if (k>1 && abs(obj_value[k] - obj_value[k-1]) <= 1e-2*obj_value[k-1])
  #   break
}
cat("Number of iterations:", k)
R>> Number of iterations: 25
plot(obj_value, type = "b", col = "blue",
     main = "Convergence of SCA", xlab = "iteration", ylab = "objective value")



# We can now plot the achieved dollar allocation:

# plot
w_all <- cbind(w_all, "RPP (SCA)" = w_SCA)
barplot(t(w_all), col = rainbow10equal[1:9],
        main = "Portfolio allocation", xlab = "stocks", ylab = "dollars", beside = TRUE, 
        legend = colnames(w_all))


# compute risk contributions
risk_all <- cbind(risk_all, 
                  "RPP (SCA)" = as.vector(w_SCA * (Sigma %*% w_SCA)))
RRC_all <- sweep(risk_all, MARGIN = 2, STATS = colSums(risk_all), FUN = "/")  # normalize each column

# plot
barplot(t(RRC_all), col = rainbow10equal[1:9],
        main = "Relative risk contribution", xlab = "stocks", ylab = "risk", beside = TRUE, 
        legend = colnames(RRC_all))

t(RRC_all)


library(riskParityPortfolio)
#?riskParityPortfolio  # to get help for the function

# use package
rpp_mu <- riskParityPortfolio(Sigma, mu = mu, lmd_mu = 5e-5, formulation = "rc-double-index")

# plot
w_all <- cbind(w_all, "RPP + mu" = rpp_mu$w)
barplot(t(w_all), col = rainbow6equal,
        main = "Portfolio allocation", xlab = "stocks", ylab = "dollars", beside = TRUE, 
        legend = colnames(w_all))
t(w_all)

# compute returns of all portfolios
(ret_all <- xts(X_lin %*% w_all, index(X_lin)))
ret_all_trn <- ret_all[1:T_trn, ]
ret_all_tst <- ret_all[-c(1:T_trn), ]

# performance in-sample
t(table.AnnualizedReturns(ret_all_trn))


{ chart.CumReturns(ret_all, main = "Cum PnL", 
                   wealth.index = TRUE, legend.loc = "topleft", colorset = rainbow6equal)
  addEventLines(xts("training", index(X_lin[T_trn])), srt=90, pos=2, lwd = 2, col = "darkblue") }


{ chart.Drawdown(ret_all, main = "Drawdown",
                 legend.loc = "bottomleft", colorset = rainbow6equal)
  addEventLines(xts("training", index(X_lin[T_trn])), srt=90, pos=2, lwd = 2, col = "darkblue") }



chart.Drawdown(ret_all_tst, main = "Drawdown during test phase", 
               legend.loc = "bottomleft", colorset = rainbow6equal)



# compute 8 different set of portfolios from different input data
w_RPP_mu_acc <- w_RPP_acc <- NULL
for (i in 1:8) {
  # sample means with random samples
  idx <- sample(1:T_trn, T_trn/2)
  mu_ <- colMeans(X_log_trn[idx, ])
  Sigma_ <- cov(X_log_trn[idx, ])
  
  # design risk-parity portfolio
  w_RPP_acc <- cbind(w_RPP_acc, 
                     riskParityPortfolio(Sigma_)$w)
  w_RPP_mu_acc <- cbind(w_RPP_mu_acc, 
                        riskParityPortfolio(Sigma_, mu = mu_, lmd_mu = 5e-5, 
                                            formulation = "rc-double-index")$w)
}
rownames(w_RPP_mu_acc) <- rownames(w_RPP_acc) <- colnames(X_lin)


barplot(t(w_RPP_acc), col = rainbow8equal,
        main = "Different realizations of RPP", 
        xlab = "stocks", ylab = "dollars", beside = TRUE)



barplot(t(w_RPP_mu_acc), col = rainbow8equal,
        main = "Different realizations of RPP with expected return", 
        xlab = "stocks", ylab = "dollars", beside = TRUE)

