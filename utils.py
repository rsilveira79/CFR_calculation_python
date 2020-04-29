import pandas as pd
import numpy as np
import json
from scipy.stats import lognorm
from tqdm import tqdm_notebook, tqdm
from statsmodels.stats.proportion import proportion_confint

## Parameters
with open("params.json", 'r') as tc:
    params = json.load(tc)

cCFRBaseline = float(params['cCFRBaseline'])
cCFREstimateRange = (float(params['cCFREstimateRange_low']),float(params['cCFREstimateRange_high']))
zmeanLow = float(params['zmeanLow'])
zmedianLow = float(params['zmedianLow'])
zmeanMid = float(params['zmeanMid'])
zmedianMid = float(params['zmedianMid'])
zmeanHigh = float(params['zmeanHigh'])
zmedianHigh = float(params['zmedianHigh'])

## Useful functions
def muTransform(zMedian):
    return np.log(zMedian)

def sigmaTransform(zMean, mu):
    return np.sqrt(2*(np.log(zMean)-mu))

def plnorm(x, mu, sigma):
    shape  = sigma
    loc    = 0
    scale  = np.exp(mu)
    return lognorm.cdf(x, shape, loc, scale)

def hospitalisation_to_death_truncated(x,mu,sigma):
    return plnorm(x + 1, mu, sigma) - plnorm(x, mu, sigma)

def hospitalisation_to_death_truncated_low(x):
    return hospitalisation_to_death_truncated(x,muLow, sigmaLow)

def hospitalisation_to_death_truncated_mid(x):
    return hospitalisation_to_death_truncated(x,muMid, sigmaMid)

def hospitalisation_to_death_truncated_high(x):
    return hospitalisation_to_death_truncated(x,muHigh, sigmaHigh)

muLow=muTransform(zmedianLow)
sigmaLow = sigmaTransform(zmeanLow, muLow)
muMid = muTransform(zmedianMid)
sigmaMid = sigmaTransform(zmeanMid, muMid)
muHigh = muTransform(zmedianHigh)
sigmaHigh = sigmaTransform(zmeanHigh, muHigh)

def calculate_underestimate(country, dataframe, delay_func):
    df = dataframe[dataframe.country==country].iloc[::-1].reset_index(drop=True)
    cumulative_known_t = 0
    for ii in range(0,len(df)):
        known_i = 0
        for jj in range(0,ii+1):
            known_jj = df['new_cases'].loc[ii-jj]*delay_func(jj)
            known_i = known_i + known_jj
        cumulative_known_t = cumulative_known_t + known_i
    cum_known_t = round(cumulative_known_t)
    nCFR = df['new_deaths'].sum()/df['new_cases'].sum()
    cCFR = df['new_deaths'].sum()/cum_known_t
    total_deaths = df['new_deaths'].sum()
    total_cases = df['new_cases'].sum()
    nCFR_UQ, nCFR_LQ =  proportion_confint(total_deaths, total_cases)
    cCFR_UQ, cCFR_LQ =  proportion_confint(total_deaths, cum_known_t)
    quantile25, quantile75 = proportion_confint(total_deaths, cum_known_t, alpha = 0.5)
    row = {
        'country': country,
        'nCFR': nCFR, 
        'cCFR':cCFR, 
        'total_deaths':total_deaths, 
        'cum_known_t':cum_known_t, 
        'total_cases':total_cases,
        'nCFR_UQ':round(nCFR_UQ,8),
        'nCFR_LQ':round(nCFR_LQ,8),
        'cCFR_UQ':round(cCFR_UQ,8),
        'cCFR_LQ':round(cCFR_LQ,8),
        'underreporting_estimate':cCFRBaseline / (100*cCFR),
        'lower':cCFREstimateRange[0]/ (100 * cCFR_UQ),
        'upper':cCFREstimateRange[1]/ (100 * cCFR_LQ),
        'quantile25':quantile25,
        'quantile75':quantile75
    }
    return row

def return_complete_df(dataframe, delay_func):
    all_countries = dataframe['country'].unique()
    rows_countries = [calculate_underestimate(c,dataframe, delay_func) for c in tqdm(all_countries)]
    new_df = pd.DataFrame(
        data =rows_countries,
        columns = [
        'country',
        'nCFR', 
        'cCFR', 
        'total_deaths', 
        'cum_known_t', 
        'total_cases',
        'nCFR_UQ',
        'nCFR_LQ',
        'cCFR_UQ',
        'cCFR_LQ',
        'underreporting_estimate',
        'lower',
        'upper',
        'quantile25',
        'quantile75'
    ])
    return new_df