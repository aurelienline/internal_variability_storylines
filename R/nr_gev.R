#! /usr/bin/env Rscript
###################################################
#
#  Return level with CI using delta-method
# 
#
#  ICdelta(A,nban,mu,sigma,xi,covar,theta)
# A    :A-year return level
# nban : nb of blocks per year
# mu: location parameter
# sigma: scale parameter
# xi   : shape parameter
# covar:  var-cov matrix of estimators
# theta: extremal index (default 1)
# conf: confidence level for the confidence interval (default 95%)
#
# out:  zinf   z   zsup
#
###################################################

ICdelta<-function(A,nban,mu,sigma,xi,covar,theta=1,conf=0.95){
  
  p<-1/(A*nban)
  yp<- -log(1-p)/theta
  
  if (xi==0) {
    z<- mu-sigma*log(yp)
    Jac<-c(1,-log(yp))
  } else {
    z<- mu-(sigma/xi)*(1-(yp)^(-xi))    
    Jac<-c(1,-(1/xi)*(1-(yp)^(-xi)),(-sigma/(xi^2))*(xi*log(yp)*(yp)^(-xi)-1+(yp)^(-xi)))
  }
  
  var<-t(Jac)%*%covar%*%Jac 
  ICinf<- z-qnorm((1+conf)/2)*sqrt(var)    
  ICsup<- z+qnorm((1+conf)/2)*sqrt(var)    
  out<-data.frame(ICinf,z,ICsup,var) 
}

ICdeltap<-function(p,mu,sigma,xi,covar)
        {
	yp<- -log(1-p)
	z<- mu-(sigma/xi)*(1-(yp)^(-xi))
	Jac<-c(1,-(1/xi)*(1-(yp)^(-xi)),(-sigma/(xi^2))*(xi*log(yp)*(yp)^(-xi)-1+(yp)^(-xi)))
        var<-t(Jac)%*%covar%*%Jac
	ICinf<- z-1.96*sqrt(var)
	ICsup<- z+1.96*sqrt(var)
	#ICinf<- z-1.0364*sqrt(var)
	#ICsup<- z+1.0364*sqrt(var)
	out<-data.frame(ICinf,z,ICsup)
        }
