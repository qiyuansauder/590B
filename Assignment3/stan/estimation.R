install.packages(c("Rcpp", "rstan"), type = "source")
install.packages("shinystan")
library(rstan)
library(shinystan)

## read in data
process <- read.csv("data/process.csv")
process <- process[,2:4]
XX <- read.csv("data/customer.csv")
XX <- cbind(1,XX[,2:5])
amount <- read.csv("data/amount.csv")
ind <- amount[,1]
t <- amount[,2]
z <- amount[,3]
R <- diag(3)
data <- list(process=process, XX=XX, ind=ind, t=t, z=z, n=dim(process)[1], 
             nn = length(ind), k = dim(XX)[2], p = 3, R=R)


## run the model; 
draws <- stan(file="model.stan", data=data, iter=2000, chains=1)
sims <- extract(draws, permuted=T)
plot(sims$beta, type="l")
summary(sims$beta)
dim(sims$B)
plot(sims$B[,1,1], type="l")
plot(sims$B[,2,1], type="l")
plot(sims$B[,1,3], type="l")

traceplot(draws,  pars="beta")
traceplot(draws,  pars="B", nrow=5)

launch_shinystan(draws)

