#! /usr/bin/env Rscript
# Sylvie Parey & Aurélien Liné

# libraries
library(ncdf4)
library(ismev)
library(evir)
library(evd)
library(lubridate)
library(stringr)

# additional codes
source('nr_gev.R')
source('p_gev.R')

#parameters
df <- read.csv('params.csv', header = TRUE, sep = ',')
variable <- paste(df$variable)
region <- df$region
temporality <- df$temporality
return_period <- df$return_period
confidence <- df$confidence
extremum <- paste(df$extremum)
refSrt <- df$refSrt; refEnd <- df$refEnd
intSrt <- df$intSrt; intEnd <- df$intEnd
#variable <- 'tas'
#region <- 'NEU'
#temporality <- 'JFM'
#return_period <- 10
#confidence <- 0.95
#extremum <- 'min'
#refSrt <- '1995'; refEnd <- '2014'
#intSrt <- '2020'; intEnd <- '2039'

# pre-traitement of variables
var_reg = paste(variable, region, sep = '')
print(variable)
if(extremum == 'max') {
  ec <-  1
} else if(extremum == 'min') {
  ec <- -1
}
refStrDate = paste(refSrt, '01-01', sep = '-')
refEndDate = paste(refEnd, '12-31', sep = '-')
intStrDate = paste(intSrt, '01-01', sep = '-')
intEndDate = paste(intEnd, '12-31', sep = '-')

# get netcdf files in the current directory
files <- list.files(path='.', pattern = '.nc', all.files = FALSE, full.names = FALSE)

print(files)

for (file_name in files) {

    if (str_detect(file_name, var_reg)) {

    cat('\n', file_name, '\n')

    file <- nc_open(file_name, write = F, readunlim = T, verbose = F)
    data <- ncvar_get(file, variable) #var_reg)
    if (variable == 'tas') {
        data <- data - 273.15
    }
 
    # print(dim(data)[1])
    if (str_detect(file_name, '1995') || dim(data)[1] == 16436) {
    date <- seq(mdy('01/01/1995'), mdy('12/31/2039'), by = 'days')
    }
    else if (str_detect(file_name, '1850') || dim(data)[1] == 69396) {
    date <- seq(mdy('01/01/1850'), mdy('12/31/2039'), by = 'days')
    }

    dataRef <- data[which(date == refStrDate):which(date == refEndDate), ]
    dat1    <- date[which(date == refStrDate):which(date == refEndDate)]
    dataInt <- data[which(date == intStrDate):which(date == intEndDate), ]
    dat2    <- date[which(date == intStrDate):which(date == intEndDate)]

    if(temporality == 'yr') {
        tb <- 365
    } else if(temporality == 'JFM') {
        dataRef <- dataRef[which(month(dat1) == 1 | month(dat1) == 2 | month(dat1) == 3),]
        dat1 <- dat1[which(month(dat1) == 1 | month(dat1) == 2 | month(dat1)==3)]
        dataInt <- dataInt[which(month(dat2) == 1 | month(dat2) == 2 | month(dat2) == 3),]
        dat2 <- dat2[which(month(dat2) == 1 | month(dat2) == 2 | month(dat2) == 3)]
        tb <- 90
    } else if(temporality == 'JJA') {
        dataRef <- dataRef[which(month(dat1) == 6 | month(dat1) == 7 | month(dat1) == 8),]
        dat1 <- dat1[which(month(dat1) == 6 | month(dat1) == 7 | month(dat1)==8)]
        dataInt <- dataInt[which(month(dat2) == 6 | month(dat2) == 7 | month(dat2) == 8),]
        dat2 <- dat2[which(month(dat2) == 6 | month(dat2) == 7 | month(dat2) == 8)]
        tb <- 92
    }

    if(temporality %in% list('yr', 'JFM')) {
        daysRef <- daysInt <- c()
        fev29_ref <- which(month(dat1) == 2 & day(dat1) == 29)
        fev29_int <- which(month(dat2) == 2 & day(dat2) == 29)

        for (i in 1:length(data[1,])) {
            V29_ref <- dataRef[,i]
            V29_ref[fev29_ref-1] <- .66 * V29_ref[fev29_ref-1] + .33 * V29_ref[fev29_ref]
            V29_ref[fev29_ref+1] <- .66 * V29_ref[fev29_ref+1] + .33 * V29_ref[fev29_ref]

            V29_int <- dataInt[,i]
            V29_int[fev29_int-1] <- .66 * V29_int[fev29_int-1] + .33 * V29_int[fev29_int]
            V29_int[fev29_int+1] <- .66 * V29_int[fev29_int+1] + .33 * V29_int[fev29_int]

            if (i == 1) {
                daysRef <- V29_ref[-fev29_ref]
                daysInt <- V29_int[-fev29_int]
            } else {
                daysRef <- c(daysRef, V29_ref[-fev29_ref])
                daysInt <- c(daysInt, V29_int[-fev29_int])
            }
    	}
    } else {
	daysRef = dataRef
	daysInt = dataInt
    }

    prob = c(0., 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.97, 0.98, 0.99, 0.995, 1.)
    qoRef = quantile(daysRef, probs = prob)
    print(round(qoRef, 3))
    qoInt = quantile(daysInt, probs = prob)
    print(round(qoInt, 3))

## extremes estimation

# reference period

    GEV_ref = gev.fit(xdat = apply(matrix(ec * daysRef, tb, length(daysRef) / tb), 2, max), show = FALSE)

    gev.diag(GEV_ref)

    mu_ref <- GEV_ref$mle[1]
    sig_ref <- GEV_ref$mle[2]
    xi_ref <- GEV_ref$mle[3]

    covar_ref <- GEV_ref$cov

    INTENSITY_ref = c()
    gevIC_ref <- ICdelta(return_period, 1, mu_ref, sig_ref, xi_ref, covar_ref, theta = 1, confidence)
    INTENSITY_ref[2] = ec * gevIC_ref$z ; INTENSITY_ref[1] = ec * gevIC_ref$ICsup ; INTENSITY_ref[3] = ec * gevIC_ref$ICinf
    cat('Reference period, intensity (return period', return_period, 'years): ', round(INTENSITY_ref, 1), '\n')

# period of interest

    GEV_int = gev.fit(xdat= apply(matrix(ec * daysInt, tb, length(daysInt) / tb), 2, max), show= FALSE)

    gev.diag(GEV_int)

    mu_int <- GEV_int$mle[1]
    sig_int <- GEV_int$mle[2]
    xi_int <- GEV_int$mle[3]

    covar_int<-GEV_int$cov

    INTENSITY_int = c()
    gevIC_int <- ICdelta(return_period, 1, mu_int, sig_int, xi_int, covar_int, theta = 1, confidence)
    INTENSITY_int[2] = ec * gevIC_int$z ; INTENSITY_int[1] = ec * gevIC_int$ICsup ; INTENSITY_int[3] = ec * gevIC_int$ICinf
    cat('Period of interest, intensity (return period', return_period, 'years):', round(INTENSITY_int, 1), '\n')

# intensity change

    delta = c()
    delta[1] = INTENSITY_int[1] - INTENSITY_ref[3]
    delta[2] = INTENSITY_int[2] - INTENSITY_ref[2]
    delta[3] = INTENSITY_int[3] - INTENSITY_ref[1]
    cat('Intensity change (return period', return_period, 'years):', round(delta[2], 1), 'C [', round(min(delta), 1), round(max(delta), 1), ']', '\n')

# frequency change

    out1 <- pdelta(ec * INTENSITY_ref[1], mu_int, sig_int, xi_int, covar_int, confidence)
    out3 <- pdelta(ec * INTENSITY_ref[3], mu_int, sig_int, xi_int, covar_int, confidence)
    extr <- 1 / (1 - c(out1$p, out1$pinf, out1$psup, out3$p, out3$pinf, out3$psup))
    #print(extr)
    pr <- c(min(extr), 1 / (1 - pdelta(ec * INTENSITY_ref[2], mu_int, sig_int, xi_int, covar_int, confidence)$p), max(extr))
    #print(pr)

    cat('Frequency change (return period', return_period, 'years):', round(pr[2]), 'years [', round(min(pr)), round(max(pr)), '], scale: multiplied by', round(return_period/pr[2], 2), '[', round(min(return_period/pr), 2), round(max(return_period/pr), 2), '] (divided by', round(pr[2]/return_period, 2), '[', round(min(pr/return_period), 2), round(max(pr/return_period), 2), '] )', '\n')

    df <- data.frame(Intensity = c(delta[2], min(delta), max(delta)),
                     Frequency = c(pr[2], min(pr), max(pr)))
    rownames(df) <- c('GEV', 'IC_inf', 'IC_sup')
    print(df)

# saving outputs

    file_out = paste(strsplit(file_name, split = '.nc')[1], '-', extremum, return_period, '.csv', sep='')
    write.csv(df, file_out)
    }
}
