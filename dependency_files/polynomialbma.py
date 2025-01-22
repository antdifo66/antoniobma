import numpy as np
import matplotlib.pyplot as plt
import emcee
from scipy.optimize import minimize
import cobaya
from cobaya.run import run
import FastMPClinpara as fmpc
from getdist import plots, MCSamples
import getdist
import matplotlib.pyplot as plt
from getdist import loadMCSamples
import numpy as np
import pandas as pd
import math as m
import matplotlib.pyplot as plt
from getdist.plots import GetDistPlotter


def createfit(noise, atrue, btrue, ctrue, index):
    #Step 1: create truth
    
    ftrue=0.5

    #Step 2: generate random data around this model
    N=50
    xp = np.sort(10*np.random.rand(N)-5)
    yerr = noise + 0.4 * np.random.rand(N)
    yp = atrue*xp*xp+btrue*xp+ctrue     #start with true y vals
    yp += yerr * np.random.randn(N)

    plt.errorbar(xp, yp, yerr=yerr, fmt=".k", capsize=0)
    x0 = np.linspace(-5, 5, 500)
    plt.plot(x0, atrue*x0*x0 + btrue*x0 + ctrue, "k", alpha=0.3, lw=3)
    plt.show()

    #####################################################################################

    #FIT PARABOLA
    def log_likelihoodpara(theta, xp, yp, yerr):
        log_f = np.log(ftrue)
        a, b, c = theta
        model = a*xp*xp + b*xp + c
        sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((yp - model) ** 2 / sigma2 + np.log(sigma2))

    #nll = lambda *args: -log_likelihoodpara(*args)
    #this is an initial guess near the true values with some offset (required for minimization)
    #initial = np.array([atrue, btrue, ctrue]) + 0.1 * np.random.randn(3) #remember to change the # in the randn to # of params 
    #soln = minimize(nll, initial, args=(xp, yp, yerr))
    #a_ml, b_ml, c_ml= soln.x

    #plt.errorbar(xp, yp, yerr=yerr, fmt=".k", capsize=0)
    #plt.plot(x0, atrue*x0*x0 + btrue*x0 + ctrue, "k", alpha=0.3, lw=3, label="truth")
    #plt.plot(x0, np.dot(np.vander(x0, 3), [a_ml, b_ml, c_ml]), ":k", label="ML")
    #plt.show()


    def log_like_distpara(a, b, c):
        log_f = np.log(ftrue)
        model = a*xp*xp + b*xp + c
        sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((yp - model) ** 2 / sigma2 + np.log(sigma2))

    def log_prior_distpara(a,b,c):
        if -0.5 < a < 0.5 and -0.5 < b < 0.5 and -2.0 < c < 0.0:
            #the prior is saying that there is a 100% chance the parameters are within the above ranges, zero outside
            return 0.0      #this is log prior, so ln(1) = 0,
        return -np.inf 

    def log_post_distpara(a, b, c):
        lp = log_prior_distpara(a, b, c)

        if not np.isfinite(lp):     #checks if the parameters are within the ranges given by the prior
            return -np.inf          #log, so this really returns a 0% chance (outside prior range)
        return lp + log_like_distpara(a, b, c)

    infopara = {
        "likelihood": {
            "log_prob": log_post_distpara
                        },
        "params": dict([
            ("a", 
                {"prior": {"min": 0.0, "max": 0.2}}
            ),
            ("b", 
                {"prior": {"min": -0.3, "max": 0.0}}
            ),
            ("c", 
                {"prior": {"min": -2.0, "max": 0.0}}
            )
                ]),
        "sampler": {
            "mcmc": {"burn_in": 200, "max_tries": 250}}, "output": "chainstriple"+str(index+3)+"/mcmcpara",
    }

    ############################################################################################

    #FIT LINE
    def log_likelihoodlin(theta, xp, yp, yerr):
        M, Blin, log_f = theta
        model = M * xp + Blin
        sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((yp - model) ** 2 / sigma2 + np.log(sigma2))

    #nlllin = lambda *args: -log_likelihoodlin(*args) #lambda is an anonymous function (called w/o a name), 
    #args is a placeholder for holding the value you want to put into the expression 
    #for example, 'lambda x: x+1' takes x, adds 1, and returns the result
    #so here, lambda takes some args and returns the negative log likelihood (not sure why negative, something to do with minimizing or log)
    #initiallin = np.array([0, 0, np.log(ftrue)]) + 0.1 * np.random.randn(3) #this is an initial guess near the true values with some offset (required for minimization)
    #soln = minimize(nlllin, initiallin, args=(xp, yp, yerr))
    #m_ml, b_mllin, log_f_ml = soln.x

    #plt.errorbar(xp, yp, yerr=yerr, fmt=".k", capsize=0)
    #plt.plot(x0, atrue*x0*x0 + btrue*x0 + ctrue, "k", alpha=0.3, lw=3, label="truth")
    #plt.plot(x0, np.dot(np.vander(x0, 2), [m_ml, b_mllin]), ":k", label="ML")
    #plt.show()

    B_low = -1.5
    B_high = 1.5
    M_low = -1.0
    M_high = 1.0

    def log_like_distlin(b, c, log_f):
        model = b * xp + c
        sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((yp - model) ** 2 / sigma2 + np.log(sigma2))


    def log_prior_distlin(b, c, log_f):
        if M_low < b < M_high and B_low < c < B_high and -1.0 < log_f < 1.0:
            #the prior is saying that there is a 100% chance the parameters are within the above ranges, zero outside
            return 0.0      #this is log prior, so ln(1) = 0,
        return -np.inf  

    def log_post_distlin(b, c, log_f):
        lp = log_prior_distlin(b, c, log_f)       #this takes the log prior using the above
    
        if not np.isfinite(lp):     #checks if the parameters are within the ranges given by the prior
            return -np.inf          #log, so this really returns a 0% chance (outside prior range)
        return lp + log_like_distlin(b, c, log_f)

    infolin = {
        "likelihood": {
            "log_prob": log_post_distlin
                    },
        "params": dict([
            ("b", 
                {"prior": {"min": M_low, "max": M_high}}
            ),
            ("c", 
                {"prior": {"min": B_low, "max": B_high}}
            ),
            ("log_f", 
                {"prior": {"min": -1.0, "max": 1.0}}
            )
                ]),
        "sampler": {
            "mcmc": {"burn_in": 200, "max_tries": 250}}, "output": "chainstriple"+str(index+3)+"/mcmclin",
    }

    ######################################################################################

    #FIT CUBIC
    def log_likelihoodcube(theta, xp, yp, yerr):
        log_f = np.log(ftrue)
        a, b, c, d  = theta
        model = a*xp*xp*xp + b*xp*xp + c*xp +d
        sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((yp - model) ** 2 / sigma2 + np.log(sigma2))

    #nll = lambda *args: -log_likelihoodcube(*args)
    #this is an initial guess near the true values with some offset (required for minimization)
    #initial = np.array([0.01, 0.31,-1.0,-1.0])  #remember to change the # in the randn to # of params 
    #soln = minimize(nll, initial, args=(xp, yp, yerr))
    #A_ml, B_ml, C_ml, D_ml= soln.x

    #plt.errorbar(xp, yp, yerr=yerr, fmt=".k", capsize=0)
    #plt.plot(x0, atrue*x0*x0 + btrue*x0 + ctrue, "k", alpha=0.3, lw=3, label="truth")

    #plt.plot(x0, np.dot(np.vander(x0, 4), [A_ml, B_ml, C_ml, D_ml]), ":k", label="ML")
    #plt.show()
    def log_like_distcube(d, a, b, c):
        log_f = np.log(ftrue)
        model = d*xp*xp*xp + a*xp*xp + b*xp + c
        sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((yp - model) ** 2 / sigma2 + np.log(sigma2))

    def log_prior_distcube(d, a, b, c):
        if 0.0 < d < 0.1 and 0.0 < a < 0.5 and -2.0 < b < 0.0 and -2.0 < c < 0.0:
            #the prior is saying that there is a 100% chance the parameters are within the above ranges, zero outside
            return 0.0      #this is log prior, so ln(1) = 0,
        return -np.inf 

    def log_post_distcube(d, a, b, c):
        lp = log_prior_distcube(d, a, b, c)

        if not np.isfinite(lp):     #checks if the parameters are within the ranges given by the prior
            return -np.inf          #log, so this really returns a 0% chance (outside prior range)
        return lp + log_like_distcube(d, a, b, c)

    infocube = {
        "likelihood": {
            "log_prob": log_post_distcube
                        },
        "params": dict([
            ("d", 
                {"prior": {"min": 0.0, "max": 0.1}}
            ),
            ("a", 
                {"prior": {"min": 0.0, "max": 0.5}}
            ),
            ("b", 
                {"prior": {"min": -2.0, "max": 0.0}}
            ),
            ("c",
                {"prior": {"min": -2.0, "max": 0.0}}
            )
                ]),
        "sampler": {
            "mcmc": {"burn_in": 200, "max_tries": 250}}, "output": "chainstriple"+str(index+3)+"/mcmccube",
    }

    #####################################################################################

    return infopara, infolin, infocube

    # here

def runmcmcs(infopara, infolin, infocube, atrue, btrue, ctrue, index):
    
    #MODEL RUNS
    updated_infopara, samplerpara = run(infopara)
    gdsamplepara = samplerpara.products(to_getdist=True)["sample"]
    updated_infocube, samplercube = run(infocube)
    gdsamplecube = samplercube.products(to_getdist=True)["sample"]
    updated_infolin, samplerlin = run(infolin)
    gdsamplelin = samplerlin.products(to_getdist=True)["sample"]

    #BMA RUN
    roots = ["\mcmcpara.1.txt", "\mcmccube.1.txt", "\mcmclin.1.txt"] # this might not be correct
    models = ["para", "cube", "lin"]
    temperature = {'para': 1, 'cube': 1, 'lin': 1}
    model_prior = {'para': 0.3, 'cube': 0.3, 'lin': 0.3}
    parameters = {'para':['a','b','c'],'cube':['d','a','b', 'c'],'lin':['b','c','log_f']}
    burnin = [0.3,0.3,0.3]
    chainpath = "C:/Users/zombk/Documents/vscode/.venv/W2025coopcode/.venv/code_for_getting_started/BMA_stuff/chainstriple"+str(index+3)+""
    out = fmpc.run(roots=roots, models=models, chainDir=chainpath, burnin=burnin,
               priors=model_prior, temperature=temperature, pars=parameters, estimator='learnt harmonic')
    
    #PLOT MODELS ONLY
    #g = plots.get_subplot_plotter()
    #g.triangle_plot([gdsamplepara, gdsamplecube, gdsamplelin], filled=False)
    #g.add_legend(['Parabola', 'Cubic', 'Linear'], colored_text=True);
    #plt.show()

    #FORMAT BMA
    outnewchains=out['newchains']['Reweighted']
    length = np.shape(outnewchains['b'])[0]
    #print(length)
    arraybma = np.zeros(shape=(length, 3))      #change 3/4 is adding/taking away weight column
    #print(arraybma)
    #print(outnewchains['a'])
    for i in range(length):
        arraybma[i][0] = outnewchains['a'][i]
        arraybma[i][1] = outnewchains['b'][i]
        arraybma[i][2] = outnewchains['c'][i]
    ## check to find NaN and tke them out
    newlength = 0
    #print(length)
    for i in range(length):
        if m.isnan(arraybma[i][0]):
            #print("break here")
            break
        newlength+=1

    arraybma = arraybma[:newlength]
    newweights = outnewchains['weight'].to_numpy() 
    newweights = newweights[:newlength]
    MCbmasamples = MCSamples(samples=arraybma, weights=newweights, names = ['a','b','c'], labels=['a','b','c'], label="BMA")

    #g = GetDistPlotter()
    # Add the sample to the plotter
    #g.triangle_plot([MCbmasamples], filled=True)
    # Show the plot
    #plt.show()


    #TRIANGLE PLOT INCLUDING BMA
    g = plots.get_subplot_plotter()
    g.triangle_plot([gdsamplepara, gdsamplecube, gdsamplelin, MCbmasamples], ['a','b','c',], filled=False, legend_labels=['Parabolic','Cubic', 'Linear','BMA'], 
                markers={'a':atrue,'b':btrue,'c':ctrue})
    plt.show()
    



#annoying, because either have to make all x0, y0, atrue, etc either global, or pass them into each one




