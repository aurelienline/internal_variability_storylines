#! /usr/bin/env Rscript
# R script that performs quantile regression using R package quantreg
# See https://cran.r-project.org/web/packages/quantreg/quantreg.pdf for
# all options and parameter values
# Note: quantreg has to be installed if not present already
# See https://cran.r-project.org/web/packages/quantreg/vignettes/rq.pdf
# By default, the significance level alpha is 0.1
#
# Author: L. Terray 15/08/2018 CERFACS
# Modified: L. Terray 12/09/2018
#          the sequence of quantiles is now an argument of the R script.
#          It is passed as a long string, then splitted as a list and unlisted
# ---------------------------------------------------------------------
args = commandArgs(trailingOnly=TRUE)
if (length(args)!=7) {
   stop("seven arguments must be supplied.n", call.=FALSE)
   }
# load library   
library(quantreg)

# Read data coming from ncl script
dat = read.csv(args[1])
clust = read.csv(args[4])
Nboot = 1000
# Define quantile sequence
tau_seq <- as.numeric(unlist(strsplit(args[6],",")))
# Calculate regression coefficient for the first quantile
# fit options: "fn" Frisch-Newton "br" (default): linear l1 norm
if ( identical(args[7],"FN") ) {
   qreg <- rq(dat$ewi ~ dat$time, tau = tau_seq[1], method="fn")
   qreg10 <- rq(dat$ewi ~ dat$time, tau = tau_seq[2], method="fn")
   qreg15 <- rq(dat$ewi ~ dat$time, tau = tau_seq[3], method="fn")
   qreg50 <- rq(dat$ewi ~ dat$time, tau = tau_seq[10], method="fn")
   qreg85 <- rq(dat$ewi ~ dat$time, tau = tau_seq[17], method="fn")
   qreg90 <- rq(dat$ewi ~ dat$time, tau = tau_seq[18], method="fn")
   qreg95 <- rq(dat$ewi ~ dat$time, tau = tau_seq[19], method="fn")
   } else if ( identical(args[7],"BR") ) {
   qreg <- rq(dat$ewi ~ dat$time, tau = tau_seq[1], method="br")
   } else {
   stop("Option for fit not implemented.n", call.=FALSE)
}
anova(qreg,qreg10,qreg15)
anova(qreg,qreg10,qreg15,joint=FALSE)
anova(qreg85,qreg90,qreg95)
anova(qreg85,qreg90,qreg95,joint=FALSE)
anova(qreg,qreg10,qreg15,qreg85,qreg90,qreg95)
anova(qreg,qreg10,qreg15,qreg85,qreg90,qreg95,joint=FALSE)

# Calculate 95% confidence interval (alpha = 0.05)
if ( identical(args[5],"RANK") ) {
   out <- summary(qreg, se="rank", alpha = 0.05, iid = FALSE)
   } else if ( identical(args[5],"BOOT") ) {
   out <- summary(qreg, se="boot", bsmethod = "cluster", cluster = clust$cluster, R = Nboot, alpha = 0.05)
   } else if ( identical(args[5],"BOOTXY") ) {
   out <- summary(qreg, se="boot", bsmethod = "xy", R = Nboot, alpha = 0.05)
   } else {
   stop("Option for CI not implemented.n", call.=FALSE)
}
# Write results (Coeff + Conf.Int) in file 
write.table(out$coefficients,file = args[2], append = FALSE, quote = TRUE,sep = " ", col.names = FALSE, row.names = FALSE)

# Loop on remaining quantiles, repeating all the above steps
for(i in 2:length(tau_seq)){
      if ( identical(args[7],"FN") ) {
      	 qreg <- rq(dat$ewi ~ dat$time, tau = tau_seq[i], method="fn")
	 } else if ( identical(args[7],"BR") ) {
	 qreg <- rq(dat$ewi ~ dat$time, tau = tau_seq[i], method="br")
	 } else {
 	   stop("Option for fit not implemented.n", call.=FALSE)
      }
      if ( identical(args[5],"RANK") ) {
      	 out <- summary(qreg, se="rank", alpha = 0.05, iid = FALSE)
	 } else if ( identical(args[5],"BOOT") ) {
  	 out <- summary(qreg, se="boot", bsmethod = "cluster", cluster = clust$cluster, R = Nboot, alpha = 0.05)
	 } else if ( identical(args[5],"BOOTXY") ) {
	 out <- summary(qreg, se="boot", bsmethod = "xy", R = Nboot, alpha = 0.05)
  	 } else {
   	 stop("Option not implemented.n", call.=FALSE)
	}	 
      write.table(out$coefficients,file = args[2], append = TRUE, quote = TRUE,sep = " ", col.names = FALSE, row.names = FALSE)
      }
# test if slope for different quantiles are significantly different

# Calculate also Ordinary least square (ols) coeff. and standard error for comparison
ols <- lm(dat$ewi ~ dat$time)
lfi <- summary(lm(dat$ewi ~ dat$time))
write.table(lfi$coefficients,file = args[3], append = FALSE, quote = TRUE,sep = " ", col.names = FALSE, row.names = FALSE)

