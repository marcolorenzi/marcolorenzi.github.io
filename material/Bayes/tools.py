import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize import minimize
from scipy.stats import norm, uniform, binom
from scipy.stats import gaussian_kde
from scipy.stats import multivariate_normal

def laplace_solution(params, other_params, data, lik, priors):
  
    def evaluate_log_post(params, other_params, data, lik, priors):
        model_list = { 'gaussian' : norm, 'uniform': uniform, 'binomial' : binom}
    
        model_lik = model_list[lik]
    
        #Computing log-priors
        log_prior = 0
        for i,mod in enumerate(priors):
            log_prior += model_list[mod[0]].logpdf(params[i], *mod[1])

            
        #Computing log-likelihood
        if lik == 'gaussian':
            # Dirty trick for guaranteeing positive variance
            params[-1] = np.abs(params[-1])
                
        if len(other_params)>0:
            log_lik = np.sum([model_list[lik].logpdf(point, *(params,other_params)) for point in data])
        else:
            log_lik = np.sum([model_list[lik].logpdf(point, *params) for point in data])
        return - (log_lik + log_prior)
    
    minimum =  minimize(evaluate_log_post, params,  
                        args = (other_params, data, lik, priors), method = 'BFGS')
    print(minimum)
    return [minimum.x, minimum.hess_inv]


def laplace_solution_regression(expression, data, lik, priors):
    model_list = { 'gaussian' : norm, 'uniform': uniform, 'binomial' : binom}
    
    def evaluate_log_post(params, var_names, data, lik, priors):
        model_list = { 'gaussian' : norm, 'uniform': uniform, 'binomial' : binom}
        model_lik = model_list[lik]
    
        #Computing log-priors
        log_prior = 0
        for i,mod in enumerate(priors):
            log_prior += model_list[mod[0]].logpdf(params[i], *mod[1])

        #Evaluating expression
        target, predictors = var_names[0], var_names[1]
        
        mu = np.ones(len(data[predictors[0]])) * params[0]
        
        for i in range(len(predictors)):
            mu += params[i+1] * data[predictors[i]].values
            
        sigma = np.abs(params[-1])
        
        t = data[target].values
        N = len(t)
               
        log_lik = np.sum([model_list['gaussian'].logpdf(t[i], mu[i], sigma) for i in range(N)])
        return -(log_lik + log_prior)
    
    collapsed_expression = expression.replace(" ", "")
    target, independent = collapsed_expression.split('~') 
    independent = independent.split('+')
    var_names = [target, independent]
    
    params = []
    for i in range(len(priors)):
        params.append(model_list[priors[i][0]].rvs(*priors[i][1]))
    
    minimum =  minimize(evaluate_log_post, params, args = (var_names, data, lik, priors), method = 'BFGS')
    print(minimum)
    return [minimum.x, minimum.hess_inv]