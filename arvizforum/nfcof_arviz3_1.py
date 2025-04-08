'''
'''
#Imports ----------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import minimize
import emcee
import getdist
from getdist import plots, MCSamples
from getdist.plots import GetDistPlotter
import math as m
import FastMPCarviz as fmpc
import arviz as az
from tqdm import tqdm
#random arviz
import numba
import importlib.resources
import functools
from arviz.stats.stats_utils import autocov as _autocov
from arviz.data.inference_data import InferenceData
from arviz.data.io_cmdstan import from_cmdstan
from arviz.data.io_beanmachine import from_beanmachine
from arviz.data.io_cmdstanpy import from_cmdstanpy
from arviz.data.io_emcee import from_emcee
from arviz.data.io_numpyro import from_numpyro
from arviz.data.io_pyro import from_pyro
from arviz.data.io_pystan import from_pystan
from arviz.data.base import dict_to_dataset
import xarray as xr
from xarray import apply_ufunc
import warnings
import re
import scipy
from scipy import stats
import packaging
from collections.abc import Sequence
from copy import copy as _copy
from copy import deepcopy as _deepcopy

try:
    from tree import is_nested
except ImportError:
    is_nested = lambda obj: False
    

#Main Program Functions ----------------------------------------
def createfit(params,noise,error,index,datarange,seed,N, a_dFix,fixpm,findelpd, myimpsamp, nologsumexp):
    '''
    This function creates randomized data from a polynomial model, and then attempts to fit a parabola,
    cubic, and quartic to it. It then runs emcee MCMC chains based on the fits, and creates a .txt file for
    each model.

    Parameters:
    -----------
    params: array, containing the values of parameters for the data generation model, sorted by increasing power of x
    noise: float, amount of noise to give the data generation
    index: int, useful if running this function multiple times in the same run, i.e. to change the noise,       
        true values, etc. (Used for Cobaya code, not necessarily required here)
    datarange: array, contains the start and end values for data generation
    seed: int, to set random seed
    N: number of points to generate


    Returns:
    -----------
    gdsamplescubic___: full posterior MCMC sample for each model, used for plotting later
    ardatacubic___: arviz InferenceData, containing likelihoods, posteriors, parameter samples, etc (thinned)
    elpds: array of elpd values calculated by me, used for pBMA calculation later
    explppds: array of lppd values calculated by me, used for stacking calculation later

    '''
    # Create the data --------------------------------------------------
    def genmodel(params,x):
        '''
        Takes parameter array, and x value, returns y value from polynomial with array parameters.
        '''
        y=0
        for i in range(len(params)):
             y+=params[i]*x**(i)
        return y

    # Generate some synthetic data from the model.
    f_true = 0.5
    np.random.seed(seed)
    if datarange[0] < 0.0:
        xp = np.sort((datarange[1]-datarange[0]) * np.random.rand(N)-((datarange[1]-datarange[0])/2))
    else:
        xp = np.sort((datarange[1]-datarange[0]) * np.random.rand(N))
    yerr = noise + error * np.random.rand(N)
    #yp = m_true * xp + b_true
    yp = genmodel(params,xp)
    #yp += np.abs(f_true * yp) * np.random.randn(N)
    yp += yerr * np.random.randn(N)
    
    plt.errorbar(xp, yp, yerr=yerr, fmt=".k", capsize=0)
    x0 = np.linspace(datarange[0], datarange[1], 500)
    plt.plot(x0, genmodel(params,x0), "k", alpha=0.3, lw=3)
    plt.xlim(datarange[0], datarange[1])
    plt.xlabel("x")
    plt.ylabel("y")


    # Fit the models ----------------------------------------------------

    # Calculate covariance matrix
        
    cov = np.diag(np.square(yerr)+0.25*np.square(genmodel(params,xp)))
    covinv = np.linalg.inv(cov)
    cinvdia = []
    for i in range(len(yerr)):
        cinvdia.append(covinv[i][i])



    #CUBIC (fid)---------------------------------------
    modeltype="cubicfid"                                                      
    
    dFixfid = a_dFix

    
    def log_likecubicfid(theta, x, y, covinv):
        a, b, c = theta                                                    
        model = dFixfid*x*x*x + a*x*x + b*x + c                                       

        diff = y - model
        chi2 = (diff.dot(covinv)).dot(diff)
        return -0.5 * chi2 

    
    nll = lambda *args: -log_likecubicfid(*args)
    initial = np.array([params[2],params[1],params[0]]) + 0.05 * np.random.randn(3)          
    solncubicfid = minimize(nll, initial, args=(xp, yp, covinv))                                        


    def log_priorcubicfid(theta):
        a,b,c = theta                                         

        alow = params[2] - 3.0
        ahigh = params[2] + 3.0
        blow = params[1] - 3.5
        bhigh = params[1] + 3.5
        clow = params[0] - 4.5
        chigh = params[0] + 4.5

        if alow < a < ahigh and blow < b < bhigh and clow < c < chigh:                     
            return np.log(1/((ahigh-alow)*(bhigh-blow)*(chigh-clow)))                        
        return -np.inf

    
    def lnprob_blobcubicfid(theta, x, y, covinv):
        prior = log_priorcubicfid(theta)
        like_vect = log_likecubicfid(theta, x, y, covinv)
        like = np.sum(like_vect)
        return like + prior, like_vect


    #run MCMC
    poscubicfid = solncubicfid.x + 1e-4 * np.random.randn(32, 3)            
    nwalkers, ndim = poscubicfid.shape
    
    samplercubicfid = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob_blobcubicfid, args=(xp, yp, covinv))
    
    samplercubicfid.run_mcmc(poscubicfid, 2000, progress=True, store=True);
    
    
    flat_samplescubicfid = samplercubicfid.get_chain(discard=100, thin = 15, flat=True)
     
    
    #steps for BMA
    def log_like_distcubicfid(a,b,c):                    
        model = dFixfid*xp*xp*xp + a*xp*xp + b*xp + c

        diff = yp - model

        chi2 = (diff.dot(covinv)).dot(diff)

        return -0.5 * chi2 



    def log_prior_distcubicfid(a,b,c):                                 
        

        alow = params[2] - 3.0
        ahigh = params[2] + 3.0
        blow = params[1] - 3.5
        bhigh = params[1] + 3.5
        clow = params[0] - 4.5
        chigh = params[0] + 4.5

        
        if alow < a < ahigh and blow < b < bhigh and clow < c < chigh:     
            return np.log(1/((ahigh-alow)*(bhigh-blow)*(chigh-clow)))     
        return -np.inf  
        
    def logpost2cubicfid(a,b,c):                   
        lp = log_prior_distcubicfid(a,b,c)       
        if not np.isfinite(lp):     
            return -np.inf, -lp        
        ll = log_like_distcubicfid(a,b,c)
        lpost = lp+ll
        return -lpost, -lp

    lpostfid = np.zeros(np.shape(flat_samplescubicfid)[0])
    lpriorfid = np.zeros(np.shape(flat_samplescubicfid)[0])
    for i in range(np.shape(flat_samplescubicfid)[0]):
        lpostfid[i], lpriorfid[i] = logpost2cubicfid(flat_samplescubicfid[i][0], flat_samplescubicfid[i][1], flat_samplescubicfid[i][2])    #add parameters from samples
    
    fullshape = (np.shape(flat_samplescubicfid)[0], np.shape(flat_samplescubicfid)[1]+3)
    fullsamplescubicfid = np.zeros(fullshape)
    
    for i in range(np.shape(flat_samplescubicfid)[0]):
        for j in range(np.shape(flat_samplescubicfid)[1]):
            fullsamplescubicfid[i][j] = flat_samplescubicfid[i][j]
            fullsamplescubicfid[i][np.shape(flat_samplescubicfid)[1]] = lpostfid[i]
            fullsamplescubicfid[i][np.shape(flat_samplescubicfid)[1]+1] = lpriorfid[i]
            fullsamplescubicfid[i][np.shape(flat_samplescubicfid)[1]+2] = 1.0

    #plot flat samples
    names=['a','b','c']                     
    
    gdsamplescubicfid = MCSamples(samples=flat_samplescubicfid,names = names, labels = ['a','b','c'])   
    
    g = plots.get_subplot_plotter()
    g.triangle_plot(gdsamplescubicfid, filled=True, markers={'a':params[2],'b':params[1],'c':params[0]})   
    
    '''
    #create txt
    fullnamescubicfid = []
    for i in range(np.shape(flat_samplescubicfid)[1]):
        fullnamescubicfid.append(names[i])
    fullnamescubicfid.append('minuslogpost')
    fullnamescubicfid.append('minuslogprior')
    fullnamescubicfid.append('weight')
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    dfcubicfid = pd.DataFrame(data=fullsamplescubicfid, columns=fullnamescubicfid)
    
    
    #this seems correct
    with open('chainar2'+modeltype+'fix.txt', mode='w') as file_object:
        print(dfcubicfid.to_string(index=False), file=file_object)
    #DONT FORGET TO INSERT A # IN THE FIRST TXT POSITION IN THE FILE
    '''

    arvizdatacubicfid = az.from_emcee(
        samplercubicfid,
        var_names=names,
        blob_names=["log_likelihood"],
        )

    arvizdatacubicfid = arvizdatacubicfid.sel(draw=slice(100, None, 15))


    #CUBIC (low)---------------------------------------
    modeltype="cubiclow"                                        
    
    dFixlow = a_dFix - fixpm
    
    def log_likecubiclow(theta, x, y, covinv):

        a, b, c = theta                                                  
        model = dFixlow*x*x*x + a*x*x + b * x + c                                         

        diff = y - model

        chi2 = (diff.dot(covinv)).dot(diff)
        return -0.5 * chi2
    
    nll = lambda *args: -log_likecubiclow(*args)
    initial = np.array([params[2],params[1],params[0]]) + 0.05 * np.random.randn(3)           
    solncubiclow = minimize(nll, initial, args=(xp, yp, covinv))


    def log_priorcubiclow(theta):
        a,b,c = theta                                                     

        alow = params[2] - 3.0
        ahigh = params[2] + 3.0
        blow = params[1] - 3.5
        bhigh = params[1] + 3.5
        clow = params[0] - 4.5
        chigh = params[0] + 4.5

        if alow < a < ahigh and blow < b < bhigh and clow < c < chigh:                     
            return np.log(1/((ahigh-alow)*(bhigh-blow)*(chigh-clow)))                         
        return -np.inf

    
    def lnprob_blobcubiclow(theta, x, y, covinv):
        prior = log_priorcubiclow(theta)
        like_vect = log_likecubiclow(theta, x, y, covinv)
        like = np.sum(like_vect)
        return like + prior, like_vect


    #run MCMC
    poscubiclow = solncubiclow.x + 1e-4 * np.random.randn(32, 3)                                          
    nwalkers, ndim = poscubiclow.shape
    
    samplercubiclow = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob_blobcubiclow, args=(xp, yp, covinv))
    
    samplercubiclow.run_mcmc(poscubiclow, 2000, progress=True, store=True)
    
    
    flat_samplescubiclow = samplercubiclow.get_chain(discard=100, thin = 15, flat=True)
    
    #steps for BMA
    def log_like_distcubiclow(a,b,c):                                      
        model = dFixlow*xp*xp*xp + a*xp*xp + b*xp + c

        diff = yp - model
        chi2 = (diff.dot(covinv)).dot(diff)
        return -0.5 * chi2

    def log_prior_distcubiclow(a,b,c):                               

        alow = params[2] - 3.0
        ahigh = params[2] + 3.0
        blow = params[1] - 3.5
        bhigh = params[1] + 3.5
        clow = params[0] - 4.5
        chigh = params[0] + 4.5

        
        if alow < a < ahigh and blow < b < bhigh and clow < c < chigh:     
            return np.log(1/((ahigh-alow)*(bhigh-blow)*(chigh-clow)))     
        return -np.inf  
        
    def logpost2cubiclow(a,b,c):                    
        lp = log_prior_distcubiclow(a,b,c)       
        if not np.isfinite(lp):     
            return -np.inf, -lp          
        ll = log_like_distcubiclow(a,b,c)
        lpost = lp+ll
        return -lpost, -lp

    lpostlow = np.zeros(np.shape(flat_samplescubiclow)[0])
    lprior = np.zeros(np.shape(flat_samplescubiclow)[0])
    for i in range(np.shape(flat_samplescubiclow)[0]):
        lpostlow[i], lprior[i] = logpost2cubiclow(flat_samplescubiclow[i][0], flat_samplescubiclow[i][1], flat_samplescubiclow[i][2])   
    
    fullshape = (np.shape(flat_samplescubiclow)[0], np.shape(flat_samplescubiclow)[1]+3)
    fullsamplescubiclow = np.zeros(fullshape)
    
    for i in range(np.shape(flat_samplescubiclow)[0]):
        for j in range(np.shape(flat_samplescubiclow)[1]):
            fullsamplescubiclow[i][j] = flat_samplescubiclow[i][j]
            fullsamplescubiclow[i][np.shape(flat_samplescubiclow)[1]] = lpostlow[i]
            fullsamplescubiclow[i][np.shape(flat_samplescubiclow)[1]+1] = lprior[i]
            fullsamplescubiclow[i][np.shape(flat_samplescubiclow)[1]+2] = 1.0

    #plot flat samples
    names=['a','b','c']                    
    
    gdsamplescubiclow = MCSamples(samples=flat_samplescubiclow,names = names, labels = ['a','b','c'])      
    
    g = plots.get_subplot_plotter()
    g.triangle_plot(gdsamplescubiclow, filled=True, markers={'a':params[2],'b':params[1],'c':params[0]})  
    
    '''
    #create txt
    
    fullnamescubiclow = []
    for i in range(np.shape(flat_samplescubiclow)[1]):
        fullnamescubiclow.append(names[i])
    fullnamescubiclow.append('minuslogpost')
    fullnamescubiclow.append('minuslogprior')
    fullnamescubiclow.append('weight')
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    dfcubiclow = pd.DataFrame(data=fullsamplescubiclow, columns=fullnamescubiclow)
    
    
    #this seems correct
    with open('chainar2'+modeltype+'fix.txt', mode='w') as file_object:
        print(dfcubiclow.to_string(index=False), file=file_object)
    #DONT FORGET TO INSERT A # IN THE FIRST TXT POSITION IN THE FILE
    '''

    arvizdatacubiclow = az.from_emcee(
        samplercubiclow,
        var_names=names,
        blob_names=["log_likelihood"],
        )

    arvizdatacubiclow = arvizdatacubiclow.sel(draw=slice(100, None, 15))


    #CUBIC (high)---------------------------------------
    modeltype="cubichigh"                                                   
    
    dFixhigh = a_dFix + fixpm
    
    def log_likecubichigh(theta, x, y, covinv):
        a, b, c = theta                                                   
        model = dFixhigh*x*x*x + a*x*x + b * x + c                                    

        diff = y - model
        chi2 = (diff.dot(covinv)).dot(diff)
        return -0.5 * chi2
    
    nll = lambda *args: -log_likecubichigh(*args)
    initial = np.array([params[2],params[1],params[0]]) + 0.05 * np.random.randn(3)            
    solncubichigh = minimize(nll, initial, args=(xp, yp, covinv))


    def log_priorcubichigh(theta):
        a,b,c = theta                                                           
        alow = params[2] - 3.0
        ahigh = params[2] + 3.0
        blow = params[1] - 3.5
        bhigh = params[1] + 3.5
        clow = params[0] - 4.5
        chigh = params[0] + 4.5

        if alow < a < ahigh and blow < b < bhigh and clow < c < chigh:                    
            return np.log(1/((ahigh-alow)*(bhigh-blow)*(chigh-clow)))                        
        return -np.inf

    
    def lnprob_blobcubichigh(theta, x, y, covinv):
        prior = log_priorcubichigh(theta)
        like_vect = log_likecubichigh(theta, x, y, covinv)
        like = np.sum(like_vect)
        return like + prior, like_vect


    #run MCMC
    poscubichigh = solncubichigh.x + 1e-4 * np.random.randn(32, 3)                                         
    nwalkers, ndim = poscubichigh.shape
    
    samplercubichigh = emcee.EnsembleSampler(
        nwalkers, ndim, lnprob_blobcubichigh, args=(xp, yp, covinv))
    
    samplercubichigh.run_mcmc(poscubichigh, 2000, progress=True, store=True);
    
    
    flat_samplescubichigh = samplercubichigh.get_chain(discard=100, thin = 15, flat=True)
    
    #steps for BMA
    def log_like_distcubichigh(a,b,c):                                      
        model = dFixhigh*xp*xp*xp + a*xp*xp + b*xp + c

        diff = yp - model

        chi2 = (diff.dot(covinv)).dot(diff)
        return -0.5 * chi2


    def log_prior_distcubichigh(a,b,c):                                   
        
        alow = params[2] - 3.0
        ahigh = params[2] + 3.0
        blow = params[1] - 3.5
        bhigh = params[1] + 3.5
        clow = params[0] - 4.5
        chigh = params[0] + 4.5

        
        if alow < a < ahigh and blow < b < bhigh and clow < c < chigh:    
            return np.log(1/((ahigh-alow)*(bhigh-blow)*(chigh-clow)))      
        return -np.inf  
        
    def logpost2cubichigh(a,b,c):                    
        lp = log_prior_distcubichigh(a,b,c)       
        if not np.isfinite(lp):     
            return -np.inf, -lp         
        ll = log_like_distcubichigh(a,b,c)
        lpost = lp+ll
        return -lpost, -lp

    lposthigh = np.zeros(np.shape(flat_samplescubichigh)[0])
    lprior = np.zeros(np.shape(flat_samplescubichigh)[0])
    for i in range(np.shape(flat_samplescubichigh)[0]):
        lposthigh[i], lprior[i] = logpost2cubichigh(flat_samplescubichigh[i][0], flat_samplescubichigh[i][1], flat_samplescubichigh[i][2])  
    
    fullshape = (np.shape(flat_samplescubichigh)[0], np.shape(flat_samplescubichigh)[1]+3)
    fullsamplescubichigh = np.zeros(fullshape)
    
    for i in range(np.shape(flat_samplescubichigh)[0]):
        for j in range(np.shape(flat_samplescubichigh)[1]):
            fullsamplescubichigh[i][j] = flat_samplescubichigh[i][j]
            fullsamplescubichigh[i][np.shape(flat_samplescubichigh)[1]] = lposthigh[i]
            fullsamplescubichigh[i][np.shape(flat_samplescubichigh)[1]+1] = lprior[i]
            fullsamplescubichigh[i][np.shape(flat_samplescubichigh)[1]+2] = 1.0

    #plot flat samples
    names=['a','b','c']  
    
    gdsamplescubichigh = MCSamples(samples=flat_samplescubichigh,names = names, labels = ['a','b','c'])     
    
    g = plots.get_subplot_plotter()
    g.triangle_plot(gdsamplescubichigh, filled=True, markers={'a':params[2],'b':params[1],'c':params[0]})  
    
    '''
    #create txt
    
    fullnamescubichigh = []
    for i in range(np.shape(flat_samplescubichigh)[1]):
        fullnamescubichigh.append(names[i])
    fullnamescubichigh.append('minuslogpost')
    fullnamescubichigh.append('minuslogprior')
    fullnamescubichigh.append('weight')
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    dfcubichigh = pd.DataFrame(data=fullsamplescubichigh, columns=fullnamescubichigh)
    
    
    #this seems correct
    with open('chainar2'+modeltype+'fix.txt', mode='w') as file_object:
        print(dfcubichigh.to_string(index=False), file=file_object)
    #DONT FORGET TO INSERT A # IN THE FIRST TXT POSITION IN THE FILE
    '''


    arvizdatacubichigh = az.from_emcee(
        samplercubichigh,
        var_names=names,
        blob_names=["log_likelihood"],
        )

    arvizdatacubichigh = arvizdatacubichigh.sel(draw=slice(100, None, 15)) 

    #Plot best fits
    plt.show()

    #####
    if findelpd:
        # Run LOO

        def PLL_fid(theta, x, y, invcov, i):
            '''
            Computes pointwise log-likelihood values to be weighted
            '''
            a,b,c=theta
            model = dFixfid*x*x*x + a*x*x + b*x + c                                 

            diff = y - model
            g = invcov.dot(diff)
            gi = g[i]
            sigii = invcov[i][i]
            return -0.5*np.log(2.0*np.pi) - 0.5*np.log(sigii) - 0.5*gi*gi/sigii
        
        def PLL_low(theta, x, y, invcov, i):
            '''
            Computes pointwise log-likelihood values to be weighted
            '''
            a,b,c=theta
            model = dFixlow*x*x*x + a*x*x + b*x + c                                    

            diff = y - model
            g = invcov.dot(diff)
            gi = g[i]
            sigii = invcov[i][i]

            return -0.5*np.log(2.0*np.pi) - 0.5*np.log(sigii) - 0.5*gi*gi/sigii
        
        def PLL_high(theta, x, y, invcov, i):
            '''
            Computes pointwise log-likelihood values to be weighted
            '''
            a,b,c=theta
            model = dFixhigh*x*x*x + a*x*x + b*x + c                    

            diff = y - model
            g = invcov.dot(diff)
            gi = g[i]
            sigii = invcov[i][i]

            return -0.5*np.log(2.0*np.pi) - 0.5*np.log(sigii) - 0.5*gi*gi/sigii


        if myimpsamp:
            def LOO_fid(x, y, invcov, chains):
                '''
                Should return the LOO posterior distribution. From Alan's LOO-PIT code
                '''
                num_data = len(y)
                #print("num data: "+str(num_data))
                num_samples = chains.shape[0]
                #print("num samples: "+str(num_samples))
                log_weights = np.zeros((num_data, num_samples))
                PLLs = np.zeros((num_data, num_samples))

                post_log_lik = lpostfid*-1
                #print("post log: "+str(post_log_lik))

                for i in tqdm(range(num_data), position = 0, leave = True):
                    """
                    This is the slowest step, but I've made it about 30% faster than the version for the paper
                    I'm sure some thinking could make it much faster, but it works correctly here
                    """

                    loo_y_mean = np.delete(y, i)
                    loo_x_obs = np.delete(x, i)
                    loo_covariance = np.delete(np.delete(invcov, i , 1), i, 0)

                    log_weights[i,:] = np.array([lnprob_blobcubicfid(chains[s], loo_x_obs, loo_y_mean, loo_covariance)[0] for s in range(num_samples)]) - post_log_lik
                    
                    PLLs[i,:] = np.array([PLL_fid(chains[s], x, y, invcov, i) for s in range(num_samples)])


                posterior = arvizdatacubicfid.posterior
                n_chains = len(posterior.chain)
                ess_p = ess(posterior, method="mean")
                reff = (
                    np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / (N-1)
                )

                print("Applying PSIS on {} data points".format(num_data))
                for i in tqdm(range(num_data), position = 0, leave = True):
                # replace each weight with the pareto-smoothed weight from this function az.psislw()
                    temp_log_psis, k = az.psislw(log_weights[i,:], reff)
                    log_weights[i,:] = temp_log_psis

                    if k > 0.7:
                        print("Data Index {} returned Pareto shape k = {}. Please check!".format(i, k))

                weightedPLLs = np.zeros((num_data, num_samples))
                plpd = np.zeros(num_data)
                weights = np.exp(log_weights)
                #print("weights: "+str(weights))
                sumweights = np.zeros(num_data)
                expPLL = np.exp(PLLs)

                for i in range(num_data):
                    for s in range(num_samples):
                        weightedPLLs[i][s] = weights[i][s]*expPLL[i][s]
                    plpd[i] = np.sum(weightedPLLs[i])
                    sumweights[i] = np.sum(weights[i])
                    plpd[i] = plpd[i]/sumweights[i]

                elpd = np.sum(np.log(plpd))
                return elpd, plpd
            
            def LOO_low(x, y, invcov, chains):
                '''
                Should return the LOO posterior distribution. From Alan's LOO-PIT code
                '''
                num_data = len(y)
                #print("num data: "+str(num_data))
                num_samples = chains.shape[0]
                #print("num samples: "+str(num_samples))
                log_weights = np.zeros((num_data, num_samples))
                PLLs = np.zeros((num_data, num_samples))

                post_log_lik = lpostlow*-1
                #print("post log: "+str(post_log_lik))

                for i in tqdm(range(num_data), position = 0, leave = True):
                    """
                    This is the slowest step, but I've made it about 30% faster than the version for the paper
                    I'm sure some thinking could make it much faster, but it works correctly here
                    """

                    loo_y_mean = np.delete(y, i)
                    loo_x_obs = np.delete(x, i)
                    loo_covariance = np.delete(np.delete(invcov, i , 1), i, 0)

                    log_weights[i,:] = np.array([lnprob_blobcubiclow(chains[s], loo_x_obs, loo_y_mean, loo_covariance)[0] for s in range(num_samples)]) - post_log_lik
                    
                    PLLs[i,:] = np.array([PLL_low(chains[s], x, y, invcov, i) for s in range(num_samples)])

                posterior = arvizdatacubiclow.posterior
                n_chains = len(posterior.chain)
                ess_p = ess(posterior, method="mean")

                reff = (
                    np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / (N-1)
                )
                print("Applying PSIS on {} data points".format(num_data))
                for i in tqdm(range(num_data), position = 0, leave = True):
                # replace each weight with the pareto-smoothed weight from this function az.psislw()
                    temp_log_psis, k = az.psislw(log_weights[i,:], reff)
                    log_weights[i,:] = temp_log_psis

                    if k > 0.7:
                        print("Data Index {} returned Pareto shape k = {}. Please check!".format(i, k))

                weightedPLLs = np.zeros((num_data, num_samples))
                plpd = np.zeros(num_data)
                weights = np.exp(log_weights)
                #print("weights: "+str(weights))
                sumweights = np.zeros(num_data)
                expPLL = np.exp(PLLs)

                for i in range(num_data):
                    for s in range(num_samples):
                        weightedPLLs[i][s] = weights[i][s]*expPLL[i][s]
                    plpd[i] = np.sum(weightedPLLs[i])
                    sumweights[i] = np.sum(weights[i])
                    plpd[i] = plpd[i]/sumweights[i]

                elpd = np.sum(np.log(plpd))
                return elpd, plpd
            
            def LOO_high(x, y, invcov, chains):
                '''
                Should return the LOO posterior distribution. From Alan's LOO-PIT code
                '''
                num_data = len(y)
                #print("num data: "+str(num_data))
                num_samples = chains.shape[0]
                #print("num samples: "+str(num_samples))
                log_weights = np.zeros((num_data, num_samples))
                PLLs = np.zeros((num_data, num_samples))

                post_log_lik = lposthigh*-1
                #print("post log: "+str(post_log_lik))

                for i in tqdm(range(num_data), position = 0, leave = True):
                    """
                    This is the slowest step, but I've made it about 30% faster than the version for the paper
                    I'm sure some thinking could make it much faster, but it works correctly here
                    """

                    loo_y_mean = np.delete(y, i)
                    loo_x_obs = np.delete(x, i)
                    loo_covariance = np.delete(np.delete(invcov, i , 1), i, 0)


                    log_weights[i,:] = np.array([lnprob_blobcubichigh(chains[s], loo_x_obs, loo_y_mean, loo_covariance)[0] for s in range(num_samples)]) - post_log_lik
                    
                    PLLs[i,:] = np.array([PLL_high(chains[s], x, y, invcov, i) for s in range(num_samples)])


                posterior = arvizdatacubichigh.posterior
                n_chains = len(posterior.chain)
                ess_p = ess(posterior, method="mean")

                reff = (
                    np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / (N-1)
                )
                print("Applying PSIS on {} data points".format(num_data))
                for i in tqdm(range(num_data), position = 0, leave = True):
                # replace each weight with the pareto-smoothed weight from this function az.psislw()
                    temp_log_psis, k = az.psislw(log_weights[i,:],reff)
                    log_weights[i,:] = temp_log_psis

                    if k > 0.7:
                        print("Data Index {} returned Pareto shape k = {}. Please check!".format(i, k))

                weightedPLLs = np.zeros((num_data, num_samples))
                plpd = np.zeros(num_data)
                weights = np.exp(log_weights)
                #print("weights: "+str(weights))
                sumweights = np.zeros(num_data)
                expPLL = np.exp(PLLs)

                for i in range(num_data):
                    for s in range(num_samples):
                        weightedPLLs[i][s] = weights[i][s]*expPLL[i][s]
                    plpd[i] = np.sum(weightedPLLs[i])
                    sumweights[i] = np.sum(weights[i])
                    plpd[i] = plpd[i]/sumweights[i]

                elpd = np.sum(np.log(plpd))
                return elpd, plpd

            elpds = np.zeros(3)
            explppds = np.zeros((3,len(yp),))

            fidout = LOO_fid(xp,yp,covinv,flat_samplescubicfid)
            lowout = LOO_low(xp,yp,covinv,flat_samplescubiclow)
            highout = LOO_high(xp,yp,covinv,flat_samplescubichigh)
            
            elpds[0] = fidout[0]
            elpds[1] = lowout[0]
            elpds[2] = highout[0]

            explppds[0,:] = fidout[1]
            explppds[1,:] = lowout[1]
            explppds[2,:] = highout[1]
        else:

            def LOO_fid(x, y, invcov, chains):
                '''
                Should return the LOO posterior distribution. From Alan's LOO-PIT code
                '''
                num_data = len(y)

                num_samples = chains.shape[0]

                inference_datafid = convert_to_inference_data(arvizdatacubicfid)
                log_likelihoodfid = _get_log_likelihood(inference_datafid)
                #print("log likelihoodfid: "+str(log_likelihoodfid))
                #same as arviz code
                log_likelihoodfid = log_likelihoodfid.stack(__sample__=("chain", "draw"))
                shape = log_likelihoodfid.shape
                    #print("second log_likelihood shape: "+str(shape))
                n_samples = shape[-1]
                n_data_points = np.prod(shape[:-1])
                
                posterior = arvizdatacubicfid.posterior
                n_chains = len(posterior.chain)
                ess_p = ess(posterior, method="mean")
                # this mean is over all data variables
                reff = (
                    np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / (N-1)
                )

                
                log_weightsfid, pareto_shapefid = az.psislw(-log_likelihoodfid, reff)
                log_weightsfid += log_likelihoodfid
                warn_mg = False
                good_k = min(1 - 1 / np.log10(n_samples), 0.7)

                if np.any(pareto_shapefid > good_k):
                    warnings.warn(
                    f"Estimated shape parameter of Pareto distribution is greater than {good_k:.2f} "
                    "for one or more samples. You should consider using a more robust model, this is "
                    "because importance sampling is less likely to work well if the marginal posterior "
                    "and LOO posterior are very different. This is more likely to happen with a "
                    "non-robust model and highly influential observations."
                        )
                    warn_mg = True

                scale_value=1
                scale = "log"
                pointwise = True
                ufunc_kwargs = {"n_dims": 1, "ravel": False}
                kwargs = {"input_core_dims": [["__sample__"]]}
                loo_lppd_i = scale_value * _wrap_xarray_ufunc(
                    _logsumexp, log_weightsfid, ufunc_kwargs=ufunc_kwargs, **kwargs
                )
                loo_lppd = loo_lppd_i.values.sum()
                loo_lppd_se = (n_data_points * np.var(loo_lppd_i.values)) ** 0.5

                lppd = np.sum(
                    _wrap_xarray_ufunc(
                        _logsumexp,
                        log_likelihoodfid,
                        func_kwargs={"b_inv": n_samples},
                        ufunc_kwargs=ufunc_kwargs,
                        **kwargs,
                    ).values
                )
                p_loo = lppd - loo_lppd / scale_value

                if not pointwise:
                    return ELPDData(
                        data=[loo_lppd, loo_lppd_se, p_loo, n_samples, n_data_points, warn_mg, scale, good_k],
                        index=[
                            "elpd_loo",
                            "se",
                            "p_loo",
                            "n_samples",
                            "n_data_points",
                            "warning",
                            "scale",
                            "good_k",
                        ],
                    )
                if np.equal(loo_lppd, loo_lppd_i).all():  # pylint: disable=no-member
                    warnings.warn(
                        "The point-wise LOO is the same with the sum LOO, please double check "
                        "the Observed RV in your model to make sure it returns element-wise logp."
                    )
                return ELPDData(
                    data=[
                        loo_lppd,
                        loo_lppd_se,
                        p_loo,
                        n_samples,
                        n_data_points,
                        warn_mg,
                        loo_lppd_i.rename("loo_i"),
                        pareto_shapefid,
                        scale,
                        good_k,
                    ],
                    index=[
                        "elpd_loo",
                        "se",
                        "p_loo",
                        "n_samples",
                        "n_data_points",
                        "warning",
                        "loo_i",
                        "pareto_k",
                        "scale",
                        "good_k",
                    ],
                )

            
            def LOO_low(x, y, invcov, chains):
                '''
                Should return the LOO posterior distribution. From Alan's LOO-PIT code
                '''
                num_data = len(y)
                num_samples = chains.shape[0]


                inference_datalow = convert_to_inference_data(arvizdatacubiclow)
                log_likelihoodlow = _get_log_likelihood(inference_datalow)
                log_likelihoodlow = log_likelihoodlow.stack(__sample__=("chain", "draw"))
                shape = log_likelihoodlow.shape
                    #print("second log_likelihood shape: "+str(shape))
                n_samples = shape[-1]
                n_data_points = np.prod(shape[:-1])
                
                posterior = arvizdatacubiclow.posterior
                n_chains = len(posterior.chain)
                ess_p = ess(posterior, method="mean")
                # this mean is over all data variables
                reff = (
                    np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / (N-1)
                )

                
                log_weightslow, pareto_shapelow = az.psislw(-log_likelihoodlow, reff)
                log_weightslow += log_likelihoodlow
                warn_mg = False
                good_k = min(1 - 1 / np.log10(n_samples), 0.7)

                if np.any(pareto_shapelow > good_k):
                    warnings.warn(
                    f"Estimated shape parameter of Pareto distribution is greater than {good_k:.2f} "
                    "for one or more samples. You should consider using a more robust model, this is "
                    "because importance sampling is less likely to work well if the marginal posterior "
                    "and LOO posterior are very different. This is more likely to happen with a "
                    "non-robust model and highly influential observations."
                        )
                    warn_mg = True

                scale_value=1
                scale = "log"
                pointwise = True
                ufunc_kwargs = {"n_dims": 1, "ravel": False}
                kwargs = {"input_core_dims": [["__sample__"]]}
                loo_lppd_i = scale_value * _wrap_xarray_ufunc(
                    _logsumexp, log_weightslow, ufunc_kwargs=ufunc_kwargs, **kwargs
                )
                loo_lppd = loo_lppd_i.values.sum()
                loo_lppd_se = (n_data_points * np.var(loo_lppd_i.values)) ** 0.5

                lppd = np.sum(
                    _wrap_xarray_ufunc(
                        _logsumexp,
                        log_likelihoodlow,
                        func_kwargs={"b_inv": n_samples},
                        ufunc_kwargs=ufunc_kwargs,
                        **kwargs,
                    ).values
                )
                p_loo = lppd - loo_lppd / scale_value

                if not pointwise:
                    return ELPDData(
                        data=[loo_lppd, loo_lppd_se, p_loo, n_samples, n_data_points, warn_mg, scale, good_k],
                        index=[
                            "elpd_loo",
                            "se",
                            "p_loo",
                            "n_samples",
                            "n_data_points",
                            "warning",
                            "scale",
                            "good_k",
                        ],
                    )
                if np.equal(loo_lppd, loo_lppd_i).all():  # pylint: disable=no-member
                    warnings.warn(
                        "The point-wise LOO is the same with the sum LOO, please double check "
                        "the Observed RV in your model to make sure it returns element-wise logp."
                    )
                return ELPDData(
                    data=[
                        loo_lppd,
                        loo_lppd_se,
                        p_loo,
                        n_samples,
                        n_data_points,
                        warn_mg,
                        loo_lppd_i.rename("loo_i"),
                        pareto_shapelow,
                        scale,
                        good_k,
                    ],
                    index=[
                        "elpd_loo",
                        "se",
                        "p_loo",
                        "n_samples",
                        "n_data_points",
                        "warning",
                        "loo_i",
                        "pareto_k",
                        "scale",
                        "good_k",
                    ],
                )

                #return elpd, plpd
            
            def LOO_high(x, y, invcov, chains):
                '''
                Should return the LOO posterior distribution. From Alan's LOO-PIT code
                '''
                num_data = len(y)

                num_samples = chains.shape[0]

                inference_datahigh = convert_to_inference_data(arvizdatacubichigh)
                log_likelihoodhigh = _get_log_likelihood(inference_datahigh)
                log_likelihoodhigh = log_likelihoodhigh.stack(__sample__=("chain", "draw"))
                shape = log_likelihoodhigh.shape

                n_samples = shape[-1]
                n_data_points = np.prod(shape[:-1])
                
                posterior = arvizdatacubichigh.posterior
                n_chains = len(posterior.chain)
                ess_p = ess(posterior, method="mean")

                reff = (
                    np.hstack([ess_p[v].values.flatten() for v in ess_p.data_vars]).mean() / (N-1)
                )

                
                log_weightshigh, pareto_shapehigh = az.psislw(-log_likelihoodhigh, reff)
                log_weightshigh += log_likelihoodhigh
                warn_mg = False
                good_k = min(1 - 1 / np.log10(n_samples), 0.7)

                if np.any(pareto_shapehigh > good_k):
                    warnings.warn(
                    f"Estimated shape parameter of Pareto distribution is greater than {good_k:.2f} "
                    "for one or more samples. You should consider using a more robust model, this is "
                    "because importance sampling is less likely to work well if the marginal posterior "
                    "and LOO posterior are very different. This is more likely to happen with a "
                    "non-robust model and highly influential observations."
                        )
                    warn_mg = True

                scale_value=1
                scale = "log"
                pointwise = True
                ufunc_kwargs = {"n_dims": 1, "ravel": False}
                kwargs = {"input_core_dims": [["__sample__"]]}
                loo_lppd_i = scale_value * _wrap_xarray_ufunc(
                    _logsumexp, log_weightshigh, ufunc_kwargs=ufunc_kwargs, **kwargs
                )
                loo_lppd = loo_lppd_i.values.sum()
                loo_lppd_se = (n_data_points * np.var(loo_lppd_i.values)) ** 0.5

                lppd = np.sum(
                    _wrap_xarray_ufunc(
                        _logsumexp,
                        log_likelihoodhigh,
                        func_kwargs={"b_inv": n_samples},
                        ufunc_kwargs=ufunc_kwargs,
                        **kwargs,
                    ).values
                )
                p_loo = lppd - loo_lppd / scale_value

                if not pointwise:
                    return ELPDData(
                        data=[loo_lppd, loo_lppd_se, p_loo, n_samples, n_data_points, warn_mg, scale, good_k],
                        index=[
                            "elpd_loo",
                            "se",
                            "p_loo",
                            "n_samples",
                            "n_data_points",
                            "warning",
                            "scale",
                            "good_k",
                        ],
                    )
                if np.equal(loo_lppd, loo_lppd_i).all():  # pylint: disable=no-member
                    warnings.warn(
                        "The point-wise LOO is the same with the sum LOO, please double check "
                        "the Observed RV in your model to make sure it returns element-wise logp."
                    )
                return ELPDData(
                    data=[
                        loo_lppd,
                        loo_lppd_se,
                        p_loo,
                        n_samples,
                        n_data_points,
                        warn_mg,
                        loo_lppd_i.rename("loo_i"),
                        pareto_shapehigh,
                        scale,
                        good_k,
                    ],
                    index=[
                        "elpd_loo",
                        "se",
                        "p_loo",
                        "n_samples",
                        "n_data_points",
                        "warning",
                        "loo_i",
                        "pareto_k",
                        "scale",
                        "good_k",
                    ],
                )


            elpds = np.zeros((3,))
            explppds = np.zeros((3,len(yp),))

            fidout = LOO_fid(xp,yp,covinv,flat_samplescubicfid)
            lowout = LOO_low(xp,yp,covinv,flat_samplescubiclow)
            highout = LOO_high(xp,yp,covinv,flat_samplescubichigh)
            
            if myimpsamp:
                elpds[0] = fidout[0]
                elpds[1] = lowout[0]
                elpds[2] = highout[0]

                explppds[0,:] = fidout[1]
                explppds[1,:] = lowout[1]
                explppds[2,:] = highout[1]
            else:
                elpds[0] = fidout["elpd_loo"]
                elpds[1] = lowout["elpd_loo"]
                elpds[2] = highout["elpd_loo"]
    ####

    plt.errorbar(xp, yp, yerr=yerr, fmt=".k", capsize=0,linewidth=0.8,zorder=0)
    plt.plot(x0, genmodel(params,x0), "k", alpha=0.3, lw=3, label="truth",zorder=5)
    plt.plot(x0, dFixfid*x0*x0*x0 + solncubicfid.x[0]*x0*x0 + solncubicfid.x[1]*x0 + solncubicfid.x[2], ":r", label="Cubic-Fid", linewidth=6,zorder=10)
    plt.plot(x0, dFixlow*x0*x0*x0 + solncubiclow.x[0]*x0*x0 + solncubiclow.x[1]*x0 + solncubiclow.x[2], ":g", label="Cubic-Low", linewidth=4,zorder=15)
    plt.plot(x0, dFixhigh*x0*x0*x0 + solncubichigh.x[0]*x0*x0 + solncubichigh.x[1]*x0 + solncubichigh.x[2], ":b", label="Cubic-High", linewidth=2,zorder=20)
    plt.legend()
    plt.show()

    print("cubic fid initial guess:")
    print(solncubicfid.x)
    print("cubic low initial guess:")
    print(solncubiclow.x)
    print("cubic high initial guess:")
    print(solncubichigh.x)

    if findelpd:
        if myimpsamp:
            return gdsamplescubiclow,gdsamplescubicfid,gdsamplescubichigh,arvizdatacubiclow,arvizdatacubicfid,arvizdatacubichigh,elpds,explppds
        else:
            return gdsamplescubiclow,gdsamplescubicfid,gdsamplescubichigh,arvizdatacubiclow,arvizdatacubicfid,arvizdatacubichigh,elpds, None
    else:
        return gdsamplescubiclow,gdsamplescubicfid,gdsamplescubichigh,arvizdatacubiclow,arvizdatacubicfid,arvizdatacubichigh, None, None



def runarviz(ardatacubiclow,ardatacubicfid,ardatacubichigh):
    ''' 
    This function takes three arviz InferenceData objects, one for each model, and returns the Pseudo-BMA and Stacking
    weights calculated by arviz.
    '''
    arvizoutstacking, log_weights = az.compare({"cubicfid":ardatacubicfid,"cubichigh":ardatacubichigh, "cubiclow":ardatacubiclow}, 
                                               ic='loo',method='stacking', useArvizELPD=True)
    az.plot_compare(arvizoutstacking)

    arvizoutpBMA, log_weights = az.compare({"cubicfid":ardatacubicfid,"cubichigh":ardatacubichigh, "cubiclow":ardatacubiclow}, 
                                           ic='loo',method='pseudo-BMA', useArvizELPD = True)
    az.plot_compare(arvizoutpBMA)


    return arvizoutpBMA['weight'], arvizoutstacking['weight'], log_weights

def calcweights_mine(elpd, explppds):
    '''
    This function takes the elpds and explppds calculated in createfits() and calculates the
    Psuedo-BMA and Stacking weights from these values.
    '''
    
    pbmaweight = np.zeros(len(elpd))
    stackweight = np.zeros(len(elpd))
    sumexpelpd = sum(np.exp(elpd))


    minelpd = min(elpd)
    unnormelpd = np.exp(elpd-minelpd)
    pbmaweight = unnormelpd / np.sum(unnormelpd)

    #stacking

    def w_fuller(weights):
        return np.concatenate((weights, [max(1.0 - np.sum(weights), 0.0)]))

    def stweight(w, n, K, explppds):
        '''
        Equation to maximize to find stacking weights

        Parameters:
        -----------
        n: int, number of datapoints
        k: int, number of models
        explppds: array of exp(pointwise log likelihoods), should be n x K
        '''
        wfull = w_fuller(w)
        out = 0
        for i in range(n):
            innersum = 0 
            for k in range(K):
                innersum += wfull[k]*explppds[k][i]
            out+=np.log(innersum)
        
        return out/n
    
    def con_sum(w):
        return np.sum(w) - 1
    
    def con_ineq1(w):
        return w
    '''
    def con_ineq2(w):
        return w[1]
    def con_ineq3(w):
        return w[2]
    
    '''
    
    cons = [{'type':'eq', 'fun': con_sum},
        {'type':'ineq', 'fun': con_ineq1}
        ]

    nll = lambda *args: -stweight(*args)
    winit = np.array([1/3,1/3,1/3])
    bounds = ((0.0,1.0) for _ in range(len(winit)))
    stackweightsoln = minimize(nll,winit,args=(np.shape(explppds)[1],3,explppds),
                               bounds=bounds,
                               constraints=cons)
    stackweight = stackweightsoln.x


    
    return pbmaweight, stackweight

def calcweights_arviz(elpds):
    '''
    This function takes the elpd values calculated in createfits() and uses them to calculate the
    Pseudo-BMA and Stacking weights in the way that arviz does it.
    '''
    '''
    #pbmaweight = np.zeros(len(elpd))
    #stackweight = np.zeros(len(elpd))
    #sumexpelpd = sum(np.exp(elpd))

    #pbma
    #for i in range(len(elpd)):
        #pbmaweight[i] = np.exp(elpd[i])/(sumexpelpd)
    #minelpd = min(elpd)
    #unnormelpd = np.exp(elpd-minelpd)
    #pbmaweight = unnormelpd / np.sum(unnormelpd)
    '''

    expelpds = np.exp(elpds)

    #stacking
    def stweight(w, expelpds):
        '''
        Equation to maximize to find stacking weights

        Parameters:
        -----------
        w: array of weights
        #n: int, number of datapoints (not needed here)
        #K: int, number of models
        explppds: array of exp(pointwise log likelihoods), should be n x K
        '''

        innersum = np.log(np.dot(expelpds, w))

        return innersum
    
    def con_sum(w):
        return np.sum(w) - 1
    
    def con_ineq1(w):
        return w

    
    cons = [{'type':'eq', 'fun': con_sum},
        {'type':'ineq', 'fun': con_ineq1},
        ]

    nll = lambda *args: -stweight(*args)
    winit = np.array([1/3,1/3,1/3])
    bounds = ((0.0,1.0) for _ in range(len(winit)))
    stackweightsoln = minimize(fun=nll,x0=winit,args=(expelpds),
                               bounds=bounds,
                               constraints=cons)
    stackweight = stackweightsoln.x


    #pbma
    minelpd = min(elpds)
    z_rv = np.exp(elpds-minelpd)
    pbmaweight = ( z_rv/ np.sum(z_rv))

    
    return pbmaweight, stackweight


    


# arviz helper functions (all pretty much exactly copied from arviz source code)
def _split_chains(ary):
    """Split and stack chains."""
    ary = np.asarray(ary)
    if len(ary.shape) <= 1:
        ary = np.atleast_2d(ary)
    _, n_draw = ary.shape
    half = n_draw // 2
    return _stack(ary[:, :half], ary[:, -half:])

def _stack(x, y):
    assert x.shape[1:] == y.shape[1:]
    return np.vstack((x, y))

def _ess_mean(ary, relative=False):
    """Compute the effective sample size for the mean."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    return _ess(_split_chains(ary), relative=relative)

def _ess(ary, relative=False):
    """Compute the effective sample size for a 2D array."""
    _numba_flag = Numba.numba_flag
    ary = np.asarray(ary, dtype=float)
    if _not_valid(ary, check_shape=False):
        return np.nan
    if (np.max(ary) - np.min(ary)) < np.finfo(float).resolution:  # pylint: disable=no-member
        return ary.size
    if len(ary.shape) < 2:
        ary = np.atleast_2d(ary)
    n_chain, n_draw = ary.shape
    acov = _autocov(ary, axis=1)
    chain_mean = ary.mean(axis=1)
    mean_var = np.mean(acov[:, 0]) * n_draw / (n_draw - 1.0)
    var_plus = mean_var * (n_draw - 1.0) / n_draw
    if n_chain > 1:
        var_plus += _numba_var(stats_variance_2d, np.var, chain_mean, axis=None, ddof=1)

    rho_hat_t = np.zeros(n_draw)
    rho_hat_even = 1.0
    rho_hat_t[0] = rho_hat_even
    rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, 1])) / var_plus
    rho_hat_t[1] = rho_hat_odd

    # Geyer's initial positive sequence
    t = 1
    while t < (n_draw - 3) and (rho_hat_even + rho_hat_odd) > 0.0:
        rho_hat_even = 1.0 - (mean_var - np.mean(acov[:, t + 1])) / var_plus
        rho_hat_odd = 1.0 - (mean_var - np.mean(acov[:, t + 2])) / var_plus
        if (rho_hat_even + rho_hat_odd) >= 0:
            rho_hat_t[t + 1] = rho_hat_even
            rho_hat_t[t + 2] = rho_hat_odd
        t += 2

    max_t = t - 2
    # improve estimation
    if rho_hat_even > 0:
        rho_hat_t[max_t + 1] = rho_hat_even
    # Geyer's initial monotone sequence
    t = 1
    while t <= max_t - 2:
        if (rho_hat_t[t + 1] + rho_hat_t[t + 2]) > (rho_hat_t[t - 1] + rho_hat_t[t]):
            rho_hat_t[t + 1] = (rho_hat_t[t - 1] + rho_hat_t[t]) / 2.0
            rho_hat_t[t + 2] = rho_hat_t[t + 1]
        t += 2

    ess = n_chain * n_draw
    tau_hat = -1.0 + 2.0 * np.sum(rho_hat_t[: max_t + 1]) + np.sum(rho_hat_t[max_t + 1 : max_t + 2])
    tau_hat = max(tau_hat, 1 / np.log10(ess))
    ess = (1 if relative else ess) / tau_hat
    if np.isnan(rho_hat_t).any():
        ess = np.nan
    return ess

def _not_valid(ary, check_nan=True, check_shape=True, nan_kwargs=None, shape_kwargs=None):
    """Validate ndarray.

    Parameters
    ----------
    ary : numpy.ndarray
    check_nan : bool
        Check if any value contains NaN.
    check_shape : bool
        Check if array has correct shape. Assumes dimensions in order (chain, draw, *shape).
        For 1D arrays (shape = (n,)) assumes chain equals 1.
    nan_kwargs : dict
        Valid kwargs are:
            axis : int,
                Defaults to None.
            how : str, {"all", "any"}
                Default to "any".
    shape_kwargs : dict
        Valid kwargs are:
            min_chains : int
                Defaults to 1.
            min_draws : int
                Defaults to 4.

    Returns
    -------
    bool
    """
    ary = np.asarray(ary)

    nan_error = False
    draw_error = False
    chain_error = False

    if check_nan:
        if nan_kwargs is None:
            nan_kwargs = {}

        isnan = np.isnan(ary)
        axis = nan_kwargs.get("axis", None)
        if nan_kwargs.get("how", "any").lower() == "all":
            nan_error = isnan.all(axis)
        else:
            nan_error = isnan.any(axis)

        if (isinstance(nan_error, bool) and nan_error) or nan_error.any():
            print("Array contains NaN-value.")

    if check_shape:
        shape = ary.shape

        if shape_kwargs is None:
            shape_kwargs = {}

        min_chains = shape_kwargs.get("min_chains", 2)
        min_draws = shape_kwargs.get("min_draws", 4)
        error_msg = f"Shape validation failed: input_shape: {shape}, "
        error_msg += f"minimum_shape: (chains={min_chains}, draws={min_draws})"

        chain_error = ((min_chains > 1) and (len(shape) < 2)) or (shape[0] < min_chains)
        draw_error = ((len(shape) < 2) and (shape[0] < min_draws)) or (
            (len(shape) > 1) and (shape[1] < min_draws)
        )

        if chain_error or draw_error:
            #_log.warning(error_msg)
            print("chain error or draw error")

    return nan_error | chain_error | draw_error

def _numba_var(numba_function, standard_numpy_func, data, axis=None, ddof=0):
    """Replace the numpy methods used to calculate variance.

    Parameters
    ----------
    numba_function : function()
        Custom numba function included in stats/stats_utils.py.

    standard_numpy_func: function()
        Standard function included in the numpy library.

    data : array.
    axis : axis along which the variance is calculated.
    ddof : degrees of freedom allowed while calculating variance.

    Returns
    -------
    array:
        variance values calculate by appropriate function for numba speedup
        if Numba is installed or enabled.

    """
    if Numba.numba_flag:
        return numba_function(data, axis=axis, ddof=ddof)
    else:
        return standard_numpy_func(data, axis=axis, ddof=ddof)
    
def numba_check():
    """Check if numba is installed."""
    numba = importlib.util.find_spec("numba")
    return numba is not None  


class Numba:
    """A class to toggle numba states."""

    numba_flag = numba_check()

    @classmethod
    def disable_numba(cls):
        """To disable numba."""
        cls.numba_flag = False

    @classmethod
    def enable_numba(cls):
        """To enable numba."""
        if numba_check():
            cls.numba_flag = True
        else:
            raise ValueError("Numba is not installed")
        

class lazy_property:  # pylint: disable=invalid-name
    """Used to load numba first time it is needed."""

    def __init__(self, fget):
        """Lazy load a property with `fget`."""
        self.fget = fget

        # copy the getter function's docstring and other attributes
        functools.update_wrapper(self, fget)

    def __get__(self, obj, cls):
        """Call the function, set the attribute."""
        if obj is None:
            return self

        value = self.fget(obj)
        setattr(obj, self.fget.__name__, value)
        return value

class maybe_numba_fn:  # pylint: disable=invalid-name
    """Wrap a function to (maybe) use a (lazy) jit-compiled version."""

    def __init__(self, function, **kwargs):
        """Wrap a function and save compilation keywords."""
        self.function = function
        kwargs.setdefault("nopython", True)
        self.kwargs = kwargs

    @lazy_property
    def numba_fn(self):
        """Memoized compiled function."""
        try:
            numba = importlib.import_module("numba")
            numba_fn = numba.jit(**self.kwargs)(self.function)
        except ImportError:
            numba_fn = self.function
        return numba_fn

    def __call__(self, *args, **kwargs):
        """Call the jitted function or normal, depending on flag."""
        if Numba.numba_flag:
            return self.numba_fn(*args, **kwargs)
        else:
            return self.function(*args, **kwargs)

def conditional_jit(_func=None, **kwargs):
    """Use numba's jit decorator if numba is installed.

    Notes
    -----
        If called without arguments  then return wrapped function.

        @conditional_jit
        def my_func():
            return

        else called with arguments

        @conditional_jit(nopython=True)
        def my_func():
            return

    """
    if _func is None:
        return lambda fn: functools.wraps(fn)(maybe_numba_fn(fn, **kwargs))
    lazy_numba = maybe_numba_fn(_func, **kwargs)
    return functools.wraps(_func)(lazy_numba)


@conditional_jit(nopython=True)
def stats_variance_1d(data, ddof=0):
    a_a, b_b = 0, 0
    for i in data:
        a_a = a_a + i
        b_b = b_b + i * i
    var = b_b / (len(data)) - ((a_a / (len(data))) ** 2)
    var = var * (len(data) / (len(data) - ddof))
    return var

def stats_variance_2d(data, ddof=0, axis=1):
    if data.ndim == 1:
        return stats_variance_1d(data, ddof=ddof)
    a_a, b_b = data.shape
    if axis == 1:
        var = np.zeros(a_a)
        for i in range(a_a):
            var[i] = stats_variance_1d(data[i], ddof=ddof)
    else:
        var = np.zeros(b_b)
        for i in range(b_b):
            var[i] = stats_variance_1d(data[:, i], ddof=ddof)

    return var

def convert_to_inference_data(obj, *, group="posterior", coords=None, dims=None, **kwargs):
    r"""Convert a supported object to an InferenceData object.

    This function sends `obj` to the right conversion function. It is idempotent,
    in that it will return arviz.InferenceData objects unchanged.

    Parameters
    ----------
    obj : dict, str, np.ndarray, xr.Dataset, pystan fit
        A supported object to convert to InferenceData:
            | InferenceData: returns unchanged
            | str: Attempts to load the cmdstan csv or netcdf dataset from disk
            | pystan fit: Automatically extracts data
            | cmdstanpy fit: Automatically extracts data
            | cmdstan csv-list: Automatically extracts data
            | emcee sampler: Automatically extracts data
            | pyro MCMC: Automatically extracts data
            | beanmachine MonteCarloSamples: Automatically extracts data
            | xarray.Dataset: adds to InferenceData as only group
            | xarray.DataArray: creates an xarray dataset as the only group, gives the
                         array an arbitrary name, if name not set
            | dict: creates an xarray dataset as the only group
            | numpy array: creates an xarray dataset as the only group, gives the
                         array an arbitrary name
    group : str
        If `obj` is a dict or numpy array, assigns the resulting xarray
        dataset to this group. Default: "posterior".
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable
    kwargs
        Rest of the supported keyword arguments transferred to conversion function.

    Returns
    -------
    InferenceData
    """
    kwargs[group] = obj
    kwargs["coords"] = coords
    kwargs["dims"] = dims

    # Cases that convert to InferenceData
    if isinstance(obj, InferenceData):
        if coords is not None or dims is not None:
            raise TypeError("Cannot use coords or dims arguments with InferenceData value.")
        return obj
    elif isinstance(obj, str):
        if obj.endswith(".csv"):
            if group == "sample_stats":
                kwargs["posterior"] = kwargs.pop(group)
            elif group == "sample_stats_prior":
                kwargs["prior"] = kwargs.pop(group)
            return from_cmdstan(**kwargs)
        else:
            if coords is not None or dims is not None:
                raise TypeError(
                    "Cannot use coords or dims arguments reading InferenceData from netcdf."
                )
            return InferenceData.from_netcdf(obj)
    elif (
        obj.__class__.__name__ in {"StanFit4Model", "CmdStanMCMC"}
        or obj.__class__.__module__ == "stan.fit"
    ):
        if group == "sample_stats":
            kwargs["posterior"] = kwargs.pop(group)
        elif group == "sample_stats_prior":
            kwargs["prior"] = kwargs.pop(group)
        if obj.__class__.__name__ == "CmdStanMCMC":
            return from_cmdstanpy(**kwargs)
        else:  # pystan or pystan3
            return from_pystan(**kwargs)
    elif obj.__class__.__name__ == "EnsembleSampler":  # ugly, but doesn't make emcee a requirement
        return from_emcee(sampler=kwargs.pop(group), **kwargs)
    elif obj.__class__.__name__ == "MonteCarloSamples":
        return from_beanmachine(sampler=kwargs.pop(group), **kwargs)
    elif obj.__class__.__name__ == "MCMC" and obj.__class__.__module__.startswith("pyro"):
        return from_pyro(posterior=kwargs.pop(group), **kwargs)
    elif obj.__class__.__name__ == "MCMC" and obj.__class__.__module__.startswith("numpyro"):
        return from_numpyro(posterior=kwargs.pop(group), **kwargs)

    # Cases that convert to xarray
    if isinstance(obj, xr.Dataset):
        dataset = obj
    elif isinstance(obj, xr.DataArray):
        if obj.name is None:
            obj.name = "x"
        dataset = obj.to_dataset()
    elif isinstance(obj, dict):
        dataset = dict_to_dataset(obj, coords=coords, dims=dims)
    elif is_nested(obj) and not isinstance(obj, (list, tuple)):
        dataset = dict_to_dataset(obj, coords=coords, dims=dims)
    elif isinstance(obj, np.ndarray):
        dataset = dict_to_dataset({"x": obj}, coords=coords, dims=dims)
    elif isinstance(obj, (list, tuple)) and isinstance(obj[0], str) and obj[0].endswith(".csv"):
        if group == "sample_stats":
            kwargs["posterior"] = kwargs.pop(group)
        elif group == "sample_stats_prior":
            kwargs["prior"] = kwargs.pop(group)
        return from_cmdstan(**kwargs)
    else:
        allowable_types = (
            "xarray dataarray",
            "xarray dataset",
            "dict",
            "pytree (if 'dm-tree' is installed)",
            "netcdf filename",
            "numpy array",
            "pystan fit",
            "emcee fit",
            "pyro mcmc fit",
            "numpyro mcmc fit",
            "cmdstan fit csv filename",
            "cmdstanpy fit",
        )
        raise ValueError(
            f'Can only convert {", ".join(allowable_types)} to InferenceData, '
            f"not {obj.__class__.__name__}"
        )

    return InferenceData(**{group: dataset})

def convert_to_dataset(obj, *, group="posterior", coords=None, dims=None):
    """Convert a supported object to an xarray dataset.

    This function is idempotent, in that it will return xarray.Dataset functions
    unchanged. Raises `ValueError` if the desired group can not be extracted.

    Note this goes through a DataInference object. See `convert_to_inference_data`
    for more details. Raises ValueError if it can not work out the desired
    conversion.

    Parameters
    ----------
    obj : dict, str, np.ndarray, xr.Dataset, pystan fit
        A supported object to convert to InferenceData:

        - InferenceData: returns unchanged
        - str: Attempts to load the netcdf dataset from disk
        - pystan fit: Automatically extracts data
        - xarray.Dataset: adds to InferenceData as only group
        - xarray.DataArray: creates an xarray dataset as the only group, gives the
          array an arbitrary name, if name not set
        - dict: creates an xarray dataset as the only group
        - numpy array: creates an xarray dataset as the only group, gives the
          array an arbitrary name

    group : str
        If `obj` is a dict or numpy array, assigns the resulting xarray
        dataset to this group.
    coords : dict[str, iterable]
        A dictionary containing the values that are used as index. The key
        is the name of the dimension, the values are the index values.
    dims : dict[str, List(str)]
        A mapping from variables to a list of coordinate names for the variable

    Returns
    -------
    xarray.Dataset
    """
    inference_data = convert_to_inference_data(obj, group=group, coords=coords, dims=dims)
    dataset = getattr(inference_data, group, None)
    if dataset is None:
        raise ValueError(
            "Can not extract {group} from {obj}! See {filename} for other "
            "conversion utilities.".format(group=group, obj=obj, filename=__file__)
        )
    return dataset

def _check_tilde_start(x):
    return bool(isinstance(x, str) and x.startswith("~"))

def _subset_list(subset, whole_list, filter_items=None, warn=True, errors="raise"):
    """Handle list subsetting (var_names, groups...) across arviz.

    Parameters
    ----------
    subset : str, list, or None
    whole_list : list
        List from which to select a subset according to subset elements and
        filter_items value.
    filter_items : {None, "like", "regex"}, optional
        If `None` (default), interpret `subset` as the exact elements in `whole_list`
        names. If "like", interpret `subset` as substrings of the elements in
        `whole_list`. If "regex", interpret `subset` as regular expressions to match
        elements in `whole_list`. A la `pandas.filter`.
    errors: {"raise", "ignore"}, optional, default="raise"
        Select either to raise or ignore the invalid names.

    Returns
    -------
    list or None
        A subset of ``whole_list`` fulfilling the requests imposed by ``subset``
        and ``filter_items``.
    """
    if subset is not None:
        if isinstance(subset, str):
            subset = [subset]

        whole_list_tilde = [item for item in whole_list if _check_tilde_start(item)]
        if whole_list_tilde and warn:
            warnings.warn(
                "ArviZ treats '~' as a negation character for selection. There are "
                "elements in `whole_list` starting with '~', {0}. Please double check"
                "your results to ensure all elements are included".format(
                    ", ".join(whole_list_tilde)
                )
            )

        excluded_items = [
            item[1:] for item in subset if _check_tilde_start(item) and item not in whole_list
        ]
        filter_items = str(filter_items).lower()
        if excluded_items:
            not_found = []

            if filter_items in {"like", "regex"}:
                for pattern in excluded_items[:]:
                    excluded_items.remove(pattern)
                    if filter_items == "like":
                        real_items = [real_item for real_item in whole_list if pattern in real_item]
                    else:
                        # i.e filter_items == "regex"
                        real_items = [
                            real_item for real_item in whole_list if re.search(pattern, real_item)
                        ]
                    if not real_items:
                        not_found.append(pattern)
                    excluded_items.extend(real_items)
            not_found.extend([item for item in excluded_items if item not in whole_list])
            if not_found:
                warnings.warn(
                    f"Items starting with ~: {not_found} have not been found and will be ignored"
                )
            subset = [item for item in whole_list if item not in excluded_items]

        elif filter_items == "like":
            subset = [item for item in whole_list for name in subset if name in item]
        elif filter_items == "regex":
            subset = [item for item in whole_list for name in subset if re.search(name, item)]

        existing_items = np.isin(subset, whole_list)
        if not np.all(existing_items) and (errors == "raise"):
            raise KeyError(f"{np.array(subset)[~existing_items]} are not present")

    return subset

def _var_names(var_names, data, filter_vars=None, errors="raise"):
    """Handle var_names input across arviz.

    Parameters
    ----------
    var_names: str, list, or None
    data : xarray.Dataset
        Posterior data in an xarray
    filter_vars: {None, "like", "regex"}, optional, default=None
        If `None` (default), interpret var_names as the real variables names. If "like",
         interpret var_names as substrings of the real variables names. If "regex",
         interpret var_names as regular expressions on the real variables names. A la
        `pandas.filter`.
    errors: {"raise", "ignore"}, optional, default="raise"
        Select either to raise or ignore the invalid names.

    Returns
    -------
    var_name: list or None
    """
    if filter_vars not in {None, "like", "regex"}:
        raise ValueError(
            f"'filter_vars' can only be None, 'like', or 'regex', got: '{filter_vars}'"
        )

    if errors not in {"raise", "ignore"}:
        raise ValueError(f"'errors' can only be 'raise', or 'ignore', got: '{errors}'")

    if var_names is not None:
        if isinstance(data, (list, tuple)):
            all_vars = []
            for dataset in data:
                dataset_vars = list(dataset.data_vars)
                for var in dataset_vars:
                    if var not in all_vars:
                        all_vars.append(var)
        else:
            all_vars = list(data.data_vars)

        all_vars_tilde = [var for var in all_vars if _check_tilde_start(var)]
        if all_vars_tilde:
            warnings.warn(
                """ArviZ treats '~' as a negation character for variable selection.
                   Your model has variables names starting with '~', {0}. Please double check
                   your results to ensure all variables are included""".format(
                    ", ".join(all_vars_tilde)
                )
            )

        try:
            var_names = _subset_list(
                var_names, all_vars, filter_items=filter_vars, warn=False, errors=errors
            )
        except KeyError as err:
            msg = " ".join(("var names:", f"{err}", "in dataset"))
            raise KeyError(msg) from err
    return var_names

class Dask:
    """Class to toggle Dask states.

    Warnings
    --------
    Dask integration is an experimental feature still in progress. It can already be used
    but it doesn't work with all stats nor diagnostics yet.
    """

    dask_flag = False
    dask_kwargs = None

    @classmethod
    def enable_dask(cls, dask_kwargs=None):
        """To enable Dask.

        Parameters
        ----------
        dask_kwargs : dict
            Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.
        """
        cls.dask_flag = True
        cls.dask_kwargs = dask_kwargs

    @classmethod
    def disable_dask(cls):
        """To disable Dask."""
        cls.dask_flag = False
        cls.dask_kwargs = None

def conditional_dask(func):
    """Conditionally pass dask kwargs to `wrap_xarray_ufunc`."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not Dask.dask_flag:
            #print("functools dask func type: "+str(type(func)))
            return func(*args, **kwargs)
        user_kwargs = kwargs.pop("dask_kwargs", None)
        if user_kwargs is None:
            user_kwargs = {}
        default_kwargs = Dask.dask_kwargs
        return func(dask_kwargs={**default_kwargs, **user_kwargs}, *args, **kwargs)

    return wrapper

def update_docstring(ufunc, func, n_output=1):
    """Update ArviZ generated ufunc docstring."""
    module = ""
    name = ""
    docstring = ""
    if hasattr(func, "__module__") and isinstance(func.__module__, str):
        module += func.__module__
    if hasattr(func, "__name__"):
        name += func.__name__
    if hasattr(func, "__doc__") and isinstance(func.__doc__, str):
        docstring += func.__doc__
    ufunc.__doc__ += "\n\n"
    if module or name:
        ufunc.__doc__ += "This function is a ufunc wrapper for "
        ufunc.__doc__ += module + "." + name
        ufunc.__doc__ += "\n"
    ufunc.__doc__ += 'Call ufunc with n_args from xarray against "chain" and "draw" dimensions:'
    ufunc.__doc__ += "\n\n"
    input_core_dims = 'tuple(("chain", "draw") for _ in range(n_args))'
    if n_output > 1:
        output_core_dims = f" tuple([] for _ in range({n_output}))"
        msg = f"xr.apply_ufunc(ufunc, dataset, input_core_dims={input_core_dims}, "
        msg += f"output_core_dims={ output_core_dims})"
    else:
        output_core_dims = ""
        msg = f"xr.apply_ufunc(ufunc, dataset, input_core_dims={input_core_dims})"
    ufunc.__doc__ += msg
    ufunc.__doc__ += "\n\n"
    ufunc.__doc__ += "For example: np.std(data, ddof=1) --> n_args=2"
    if docstring:
        ufunc.__doc__ += "\n\n"
        ufunc.__doc__ += module
        ufunc.__doc__ += name
        ufunc.__doc__ += " docstring:"
        ufunc.__doc__ += "\n\n"
        ufunc.__doc__ += docstring

def make_ufunc(
    func, n_dims=2, n_output=1, n_input=1, index=Ellipsis, ravel=True, check_shape=None
):  # noqa: D202
    """Make ufunc from a function taking 1D array input.

    Parameters
    ----------
    func : callable
    n_dims : int, optional
        Number of core dimensions not broadcasted. Dimensions are skipped from the end.
        At minimum n_dims > 0.
    n_output : int, optional
        Select number of results returned by `func`.
        If n_output > 1, ufunc returns a tuple of objects else returns an object.
    n_input : int, optional
        Number of **array** inputs to func, i.e. ``n_input=2`` means that func is called
        with ``func(ary1, ary2, *args, **kwargs)``
    index : int, optional
        Slice ndarray with `index`. Defaults to `Ellipsis`.
    ravel : bool, optional
        If true, ravel the ndarray before calling `func`.
    check_shape: bool, optional
        If false, do not check if the shape of the output is compatible with n_dims and
        n_output. By default, True only for n_input=1. If n_input is larger than 1, the last
        input array is used to check the shape, however, shape checking with multiple inputs
        may not be correct.

    Returns
    -------
    callable
        ufunc wrapper for `func`.
    """
    if n_dims < 1:
        raise TypeError("n_dims must be one or higher.")

    if n_input == 1 and check_shape is None:
        check_shape = True
    elif check_shape is None:
        check_shape = False

    def _ufunc(*args, out=None, out_shape=None, **kwargs):
        """General ufunc for single-output function."""
        arys = args[:n_input]
        n_dims_out = None
        if out is None:
            if out_shape is None:
                out = np.empty(arys[-1].shape[:-n_dims])
            else:
                out = np.empty((*arys[-1].shape[:-n_dims], *out_shape))
                n_dims_out = -len(out_shape)
        elif check_shape:
            if out.shape != arys[-1].shape[:-n_dims]:
                msg = f"Shape incorrect for `out`: {out.shape}."
                msg += f" Correct shape is {arys[-1].shape[:-n_dims]}"
                raise TypeError(msg)
        for idx in np.ndindex(out.shape[:n_dims_out]):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            out_idx = np.asarray(func(*arys_idx, *args[n_input:], **kwargs))[index]
            if n_dims_out is None:
                out_idx = out_idx.item()
            out[idx] = out_idx
        return out
    def _multi_ufunc(*args, out=None, out_shape=None, **kwargs):
        """General ufunc for multi-output function."""
        arys = args[:n_input]
        element_shape = arys[-1].shape[:-n_dims]
        if out is None:
            if out_shape is None:
                out = tuple(np.empty(element_shape) for _ in range(n_output))
            else:
                out = tuple(np.empty((*element_shape, *out_shape[i])) for i in range(n_output))

        elif check_shape:
            raise_error = False
            correct_shape = tuple(element_shape for _ in range(n_output))
            if isinstance(out, tuple):
                out_shape = tuple(item.shape for item in out)
                if out_shape != correct_shape:
                    raise_error = True
            else:
                raise_error = True
                out_shape = "not tuple, type={type(out)}"
            if raise_error:
                msg = f"Shapes incorrect for `out`: {out_shape}."
                msg += f" Correct shapes are {correct_shape}"
                raise TypeError(msg)
        for idx in np.ndindex(element_shape):
            arys_idx = [ary[idx].ravel() if ravel else ary[idx] for ary in arys]
            results = func(*arys_idx, *args[n_input:], **kwargs)
            for i, res in enumerate(results):
                out[i][idx] = np.asarray(res)[index]
        return out

    if n_output > 1:
        ufunc = _multi_ufunc
    else:
        ufunc = _ufunc

    update_docstring(ufunc, func, n_output)
    return ufunc

@conditional_dask
def _wrap_xarray_ufunc(
    ufunc,
    *datasets,
    ufunc_kwargs=None,
    func_args=None,
    func_kwargs=None,
    dask_kwargs=None,
    **kwargs,
):
    """Wrap make_ufunc with xarray.apply_ufunc.

    Parameters
    ----------
    ufunc : callable
    *datasets : xarray.Dataset
    ufunc_kwargs : dict
        Keyword arguments passed to `make_ufunc`.
            - 'n_dims', int, by default 2
            - 'n_output', int, by default 1
            - 'n_input', int, by default len(datasets)
            - 'index', slice, by default Ellipsis
            - 'ravel', bool, by default True
    func_args : tuple
        Arguments passed to 'ufunc'.
    func_kwargs : dict
        Keyword arguments passed to 'ufunc'.
            - 'out_shape', int, by default None
    dask_kwargs : dict
        Dask related kwargs passed to :func:`xarray:xarray.apply_ufunc`.
        Use ``enable_dask`` method of :class:`arviz.Dask` to set default kwargs.
    **kwargs
        Passed to :func:`xarray.apply_ufunc`.

    Returns
    -------
    xarray.Dataset
    """
    #print("_wrap_xarray_ufunc")
    if ufunc_kwargs is None:
        ufunc_kwargs = {}
    ufunc_kwargs.setdefault("n_input", len(datasets))
    if func_args is None:
        func_args = tuple()
    if func_kwargs is None:
        func_kwargs = {}
    if dask_kwargs is None:
        dask_kwargs = {}

    kwargs.setdefault(
        "input_core_dims", tuple(("chain", "draw") for _ in range(len(func_args) + len(datasets)))
    )
    ufunc_kwargs.setdefault("n_dims", len(kwargs["input_core_dims"][-1]))
    kwargs.setdefault("output_core_dims", tuple([] for _ in range(ufunc_kwargs.get("n_output", 1))))

    callable_ufunc = make_ufunc(ufunc, **ufunc_kwargs)
    #print("callable ufunc type: "+str(type(callable_ufunc)))
    # none type
    return apply_ufunc(
        callable_ufunc, *datasets, *func_args, kwargs=func_kwargs, **dask_kwargs, **kwargs
    )

def _backtransform_ranks(arr, c=3 / 8):  # pylint: disable=invalid-name
    """Backtransformation of ranks.

    Parameters
    ----------
    arr : np.ndarray
        Ranks array
    c : float
        Fractional offset. Defaults to c = 3/8 as recommended by Blom (1958).

    Returns
    -------
    np.ndarray

    References
    ----------
    Blom, G. (1958). Statistical Estimates and Transformed Beta-Variables. Wiley; New York.
    """
    arr = np.asarray(arr)
    size = arr.size
    return (arr - c) / (size - 2 * c + 1)

def _z_scale(ary):
    """Calculate z_scale.

    Parameters
    ----------
    ary : np.ndarray

    Returns
    -------
    np.ndarray
    """
    ary = np.asarray(ary)
    if packaging.version.parse(scipy.__version__) < packaging.version.parse("1.10.0.dev0"):
        rank = stats.rankdata(ary, method="average")
    else:
        # the .ravel part is only needed to overcom a bug in scipy 1.10.0.rc1
        rank = stats.rankdata(  # pylint: disable=unexpected-keyword-arg
            ary, method="average", nan_policy="omit"
        )
    rank = _backtransform_ranks(rank)
    z = stats.norm.ppf(rank)
    z = z.reshape(ary.shape)
    return z

def _ess_bulk(ary, relative=False):
    """Compute the effective sample size for the bulk."""
    ary = np.asarray(ary)
    if _not_valid(ary, shape_kwargs=dict(min_draws=4, min_chains=1)):
        return np.nan
    z_scaled = _z_scale(_split_chains(ary))
    ess_bulk = _ess(z_scaled, relative=relative)
    return ess_bulk

def ess(
    data,
    *,
    var_names=None,
    method="bulk",
    relative=False,
    prob=None,
    dask_kwargs=None,
):
    r"""Calculate estimate of the effective sample size (ess).

    Parameters
    ----------
    data : obj
        Any object that can be converted to an :class:`arviz.InferenceData` object.
        Refer to documentation of :func:`arviz.convert_to_dataset` for details.
        For ndarray: shape = (chain, draw).
        For n-dimensional ndarray transform first to dataset with :func:`arviz.convert_to_dataset`.
    var_names : str or list of str
        Names of variables to include in the return value Dataset.
    method : str, optional, default "bulk"
        Select ess method. Valid methods are:

        - "bulk"
        - "tail"     # prob, optional
        - "quantile" # prob
        - "mean" (old ess)
        - "sd"
        - "median"
        - "mad" (mean absolute deviance)
        - "z_scale"
        - "folded"
        - "identity"
        - "local"
    relative : bool
        Return relative ess
        ``ress = ess / n``
    prob : float, or tuple of two floats, optional
        probability value for "tail", "quantile" or "local" ess functions.
    dask_kwargs : dict, optional
        Dask related kwargs passed to :func:`~arviz.wrap_xarray_ufunc`.

    Returns
    -------
    xarray.Dataset
        Return the effective sample size, :math:`\hat{N}_{eff}`

    Notes
    -----
    The basic ess (:math:`N_{\mathit{eff}}`) diagnostic is computed by:

    .. math:: \hat{N}_{\mathit{eff}} = \frac{MN}{\hat{\tau}}

    .. math:: \hat{\tau} = -1 + 2 \sum_{t'=0}^K \hat{P}_{t'}

    where :math:`M` is the number of chains, :math:`N` the number of draws,
    :math:`\hat{\rho}_t` is the estimated _autocorrelation at lag :math:`t`, and
    :math:`K` is the last integer for which :math:`\hat{P}_{K} = \hat{\rho}_{2K} +
    \hat{\rho}_{2K+1}` is still positive.

    The current implementation is similar to Stan, which uses Geyer's initial monotone sequence
    criterion (Geyer, 1992; Geyer, 2011).

    References
    ----------
    * Vehtari et al. (2021). Rank-normalization, folding, and
        localization: An improved Rhat for assessing convergence of
        MCMC. Bayesian analysis, 16(2):667-718.
    * https://mc-stan.org/docs/reference-manual/analysis.html#effective-sample-size.section
    * Gelman et al. BDA3 (2013) Formula 11.8

    See Also
    --------
    arviz.rhat : Compute estimate of rank normalized splitR-hat for a set of traces.
    arviz.mcse : Calculate Markov Chain Standard Error statistic.
    plot_ess : Plot quantile, local or evolution of effective sample sizes (ESS).
    arviz.summary : Create a data frame with summary statistics.

    Examples
    --------
    Calculate the effective_sample_size using the default arguments:

    .. ipython::

        In [1]: import arviz as az
           ...: data = az.load_arviz_data('non_centered_eight')
           ...: az.ess(data)

    Calculate the ress of some of the variables

    .. ipython::

        In [1]: az.ess(data, relative=True, var_names=["mu", "theta_t"])

    Calculate the ess using the "tail" method, leaving the `prob` argument at its default
    value.

    .. ipython::

        In [1]: az.ess(data, method="tail")

    """
    methods = {
        "mean": _ess_mean,
        "bulk": _ess_bulk,
    }
    #print("ess")

    if method not in methods:
        raise TypeError(f"ess method {method} not found. Valid methods are:\n{', '.join(methods)}")
    ess_func = methods[method]

    if (method == "quantile") and prob is None:
        raise TypeError("Quantile (prob) information needs to be defined.")

    if isinstance(data, np.ndarray):
        data = np.atleast_2d(data)
        if len(data.shape) < 3:
            if prob is not None:
                return ess_func(  # pylint: disable=unexpected-keyword-arg
                    data, prob=prob, relative=relative
                )

            return ess_func(data, relative=relative)

        msg = (
            "Only uni-dimensional ndarray variables are supported."
            " Please transform first to dataset with `az.convert_to_dataset`."
        )
        raise TypeError(msg)

    dataset = convert_to_dataset(data, group="posterior")
    var_names = _var_names(var_names, dataset)

    dataset = dataset if var_names is None else dataset[var_names]

    ufunc_kwargs = {"ravel": False}
    func_kwargs = {"relative": relative} if prob is None else {"prob": prob, "relative": relative}
    #print("ess_func type: "+str(type(ess_func)))
    #/\ class function
    return _wrap_xarray_ufunc(
        ess_func,
        dataset,
        ufunc_kwargs=ufunc_kwargs,
        func_kwargs=func_kwargs,
        dask_kwargs=dask_kwargs,
    )
        
def _get_log_likelihood(idata, var_name=None, single_var=True):
    """Retrieve the log likelihood dataarray of a given variable."""
    if (
        not hasattr(idata, "log_likelihood")
        and hasattr(idata, "sample_stats")
        and hasattr(idata.sample_stats, "log_likelihood")
    ):
        warnings.warn(
            "Storing the log_likelihood in sample_stats groups has been deprecated",
            DeprecationWarning,
        )
        return idata.sample_stats.log_likelihood
    if not hasattr(idata, "log_likelihood"):
        raise TypeError("log likelihood not found in inference data object")
    if var_name is None:
        var_names = list(idata.log_likelihood.data_vars)
        if len(var_names) > 1:
            if single_var:
                raise TypeError(
                    f"Found several log likelihood arrays {var_names}, var_name cannot be None"
                )
            return idata.log_likelihood[var_names]
        return idata.log_likelihood[var_names[0]]
    else:
        try:
            log_likelihood = idata.log_likelihood[var_name]
        except KeyError as err:
            raise TypeError(f"No log likelihood data named {var_name} found") from err
        return log_likelihood

def _logsumexp(ary, *, b=None, b_inv=None, axis=None, keepdims=False, out=None, copy=True):
    """Stable logsumexp when b >= 0 and b is scalar.

    b_inv overwrites b unless b_inv is None.
    """
    # check dimensions for result arrays
    ary = np.asarray(ary)
    if ary.dtype.kind == "i":
        ary = ary.astype(np.float64)
    dtype = ary.dtype.type
    shape = ary.shape
    shape_len = len(shape)
    if isinstance(axis, Sequence):
        axis = tuple(axis_i if axis_i >= 0 else shape_len + axis_i for axis_i in axis)
        agroup = axis
    else:
        axis = axis if (axis is None) or (axis >= 0) else shape_len + axis
        agroup = (axis,)
    shape_max = (
        tuple(1 for _ in shape)
        if axis is None
        else tuple(1 if i in agroup else d for i, d in enumerate(shape))
    )
    # create result arrays
    if out is None:
        if not keepdims:
            out_shape = (
                tuple()
                if axis is None
                else tuple(d for i, d in enumerate(shape) if i not in agroup)
            )
        else:
            out_shape = shape_max
        out = np.empty(out_shape, dtype=dtype)
    if b_inv == 0:
        return np.full_like(out, np.inf, dtype=dtype) if out.shape else np.inf
    if b_inv is None and b == 0:
        return np.full_like(out, -np.inf) if out.shape else -np.inf
    ary_max = np.empty(shape_max, dtype=dtype)
    # calculations
    ary.max(axis=axis, keepdims=True, out=ary_max)
    if copy:
        ary = ary.copy()
    ary -= ary_max
    np.exp(ary, out=ary)
    ary.sum(axis=axis, keepdims=keepdims, out=out)
    np.log(out, out=out)
    if b_inv is not None:
        ary_max -= np.log(b_inv)
    elif b:
        ary_max += np.log(b)
    out += ary_max if keepdims else ary_max.squeeze()
    # transform to scalar if possible
    return out if out.shape else dtype(out)
    
SCALE_DICT = {"deviance": "deviance", "log": "elpd", "negative_log": "-elpd"}

BASE_FMT = """Computed from {{n_samples}} posterior samples and \
{{n_points}} observations log-likelihood matrix.

{{0:{0}}} Estimate       SE
{{scale}}_{{kind}} {{1:8.2f}}  {{2:7.2f}}
p_{{kind:{1}}} {{3:8.2f}}        -"""
POINTWISE_LOO_FMT = """------

Pareto k diagnostic values:
                         {{0:>{0}}} {{1:>6}}
(-Inf, {{8:.2f}}]   (good)     {{2:{0}d}} {{5:6.1f}}%
   ({{8:.2f}}, 1]   (bad)      {{3:{0}d}} {{6:6.1f}}%
   (1, Inf)   (very bad) {{4:{0}d}} {{7:6.1f}}%
"""

class ELPDData(pd.Series):  # pylint: disable=too-many-ancestors
    """Class to contain the data from elpd information criterion like waic or loo."""

    def __str__(self):
        """Print elpd data in a user friendly way."""
        kind = self.index[0].split("_")[1]

        if kind not in ("loo", "waic"):
            raise ValueError("Invalid ELPDData object")

        scale_str = SCALE_DICT[self["scale"]]
        padding = len(scale_str) + len(kind) + 1
        base = BASE_FMT.format(padding, padding - 2)
        base = base.format(
            "",
            kind=kind,
            scale=scale_str,
            n_samples=self.n_samples,
            n_points=self.n_data_points,
            *self.values,
        )

        if self.warning:
            base += "\n\nThere has been a warning during the calculation. Please check the results."

        if kind == "loo" and "pareto_k" in self:
            bins = np.asarray([-np.inf, self.good_k, 1, np.inf])
            counts, *_ = _histogram(self.pareto_k.values, bins)
            extended = POINTWISE_LOO_FMT.format(max(4, len(str(np.max(counts)))))
            extended = extended.format(
                "Count",
                "Pct.",
                *[*counts, *(counts / np.sum(counts) * 100)],
                self.good_k,
            )
            base = "\n".join([base, extended])
        return base

    def __repr__(self):
        """Alias to ``__str__``."""
        return self.__str__()

    def copy(self, deep=True):  # pylint:disable=overridden-final-method
        """Perform a pandas deep copy of the ELPDData plus a copy of the stored data."""
        copied_obj = pd.Series.copy(self)
        for key in copied_obj.keys():
            if deep:
                copied_obj[key] = _deepcopy(copied_obj[key])
            else:
                copied_obj[key] = _copy(copied_obj[key])
        return ELPDData(copied_obj)


def _histogram(data, bins, range_hist=None):
    """Conditionally jitted histogram.

    Parameters
    ----------
    data : array-like
        Input data. Passed as first positional argument to ``np.histogram``.
    bins : int or array-like
        Passed as keyword argument ``bins`` to ``np.histogram``.
    range_hist : (float, float), optional
        Passed as keyword argument ``range`` to ``np.histogram``.

    Returns
    -------
    hist : array
        The number of counts per bin.
    density : array
        The density corresponding to each bin.
    bin_edges : array
        The edges of the bins used.
    """
    hist, bin_edges = np.histogram(data, bins=bins, range=range_hist)
    hist_dens = hist / (hist.sum() * np.diff(bin_edges))
    return hist, hist_dens, bin_edges

def _ic_matrix(ics, ic_i):
    """Store the previously computed pointwise predictive accuracy values (ics) in a 2D matrix."""
    cols, _ = ics.shape
    rows = len(ics[ic_i].iloc[0])
    ic_i_val = np.zeros((rows, cols))

    for idx, val in enumerate(ics.index):
        ic = ics.loc[val][ic_i]

        if len(ic) != rows:
            raise ValueError("The number of observations should be the same across all models")

        ic_i_val[:, idx] = ic

    return rows, cols, ic_i_val



    













































    



    
