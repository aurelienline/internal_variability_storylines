###################################################
#
#  probability with CI using delta-method
# 
#
#  pdelta(z,mu,sigma,xi,covar,conf)
# z : la valeur dont on cherche la probabilite
# mu: location parameter
# sigma: scale parameter
# xi   : shape parameter
# covar:  var-cov matrix of estimators
# conf: confidence level for the confidence interval (default 95%)
#
# out:  pinf   p   psup
#
###################################################

pdelta<-function(z,mu,sigma,xi,covar,conf=0.95){
  
  if (xi==0) {
    p<- exp(-exp(-(z-mu)/sigma))
    Jac<-c(-1/sigma*exp(-(z-mu)/sigma)*p,-(z-mu)/(sigma^2)*exp(-(z-mu)/sigma)*p)
  } else {
    p<- exp(-(1+xi/sigma*(z-mu))^(-1/xi))
    dpdmu <- -1/sigma*(1+xi/sigma*(z-mu))^(-1/xi-1)*p
    dpdsig <- -xi/(sigma^2)*(z-mu)*1/xi*(1+xi/sigma*(z-mu))^(-1/xi-1)*p
    dpdxi <- -(1/(xi^2)*log(1+xi/sigma*(z-mu))-1/xi*(z-mu)/(sigma+xi*(z-mu)))*(1+xi/sigma*(z-mu))^(-1/xi)*p
    Jac<-c(dpdmu,dpdsig,dpdxi)
  }

  var<-t(Jac)%*%covar%*%Jac 
  pinf<- p-qnorm((1+conf)/2)*sqrt(var)    
  psup<- p+qnorm((1+conf)/2)*sqrt(var)    
  out<-data.frame(pinf,p,psup) 
}
