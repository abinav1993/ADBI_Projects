# Collaborators:
#1. Prashanth Rallapalli
#2. Shiv shankar Barai

require("vars")
require("ucra")
# Read the input data
setwd("C:/Users/Abinav/Google Drive/2nd Semester/BI/Projects/Project - 7/causality")
data <- read.csv("data.csv")

# Build a VAR model 
# Select the lag order using the Schwarz Information Criterion with a maximum lag of 10
# see ?VARSelect to find the optimal number of lags and use it as input to VAR()
model <- VAR(data, lag.max = 10, ic = "SC")

# Extract the residuals from the VAR model 
# see ?residuals
residuals.move <- residuals(model$varresult$Move)
residuals.rprice <- residuals(model$varresult$RPRICE)
residuals.mprice <- residuals(model$varresult$MPRICE)
# Check for stationarity using the Augmented Dickey-Fuller test 
# see ?ur.df
df.move <- ur.df(residuals.move)
df.rprice <- ur.df(residuals.rprice)
df.mprice <- ur.df(residuals.mprice)
# Check whether the variables follow a Gaussian distribution  
# see ?ks.test
ks.test(residuals.move, "pnorm")
ks.test(residuals.rprice, "pnorm")
ks.test(residuals.mprice, "pnorm")
# Write the residuals to a csv file to build causal graphs using Tetrad software
residuals <- data.frame(residuals.move,residuals.rprice,residuals.mprice)
colnames(residuals) <- c("Move", "RPRICE", "MPRICE")
write.csv(residuals, file = "residuals.csv", row.names = FALSE)

