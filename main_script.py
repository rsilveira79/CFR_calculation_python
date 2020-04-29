import io
import json
import requests
import pandas as pd
import numpy as np
from scipy.stats import lognorm
from tqdm import tqdm_notebook
from statsmodels.stats.proportion import proportion_confint
from datetime import date, datetime
from utils import muTransform, sigmaTransform, plnorm
from utils import hospitalisation_to_death_truncated,hospitalisation_to_death_truncated_low, hospitalisation_to_death_truncated_mid, hospitalisation_to_death_truncated_high
from utils import calculate_underestimate, return_complete_df

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

if __name__ == "__main__":
    print(" 1 -- Starting to evaluate CFR")
    muLow=muTransform(zmedianLow)
    sigmaLow = sigmaTransform(zmeanLow, muLow)
    muMid = muTransform(zmedianMid)
    sigmaMid = sigmaTransform(zmeanMid, muMid)  
    muHigh = muTransform(zmedianHigh)
    sigmaHigh = sigmaTransform(zmeanHigh, muHigh)
    print(" 2 -- Loading dataset")
    url="https://opendata.ecdc.europa.eu/covid19/casedistribution/csv"
    s=requests.get(url).content
    dataset=pd.read_csv(io.StringIO(s.decode('utf-8')))
    dataset.rename(columns = {
    "dateRep": "date",
    "cases": "new_cases",
    "deaths": "new_deaths",
    "countriesAndTerritories": "country"},inplace=True)
    allTogetherClean = dataset[['date', 'country', 'new_cases', 'new_deaths']]
    exclude_coutries = ['Canada','Cases_on_an_international_conveyance_Japan']
    allTogetherClean = allTogetherClean[~allTogetherClean.country.isin(exclude_coutries)]
    ## Remove lower data points
    threshold = 10
    list_filtered_countried = allTogetherClean.groupby('country').filter(lambda x: x['new_deaths'].sum()>threshold)['country'].unique()
    allTogetherClean = allTogetherClean[allTogetherClean.country.isin(list_filtered_countried)].reset_index(drop=True)
    print(" 3 -- Calculating Under-Estimation")
    allTogetherLow = return_complete_df(allTogetherClean, hospitalisation_to_death_truncated_low)
    allTogetherMid = return_complete_df(allTogetherClean, hospitalisation_to_death_truncated_mid)
    allTogetherHigh = return_complete_df(allTogetherClean, hospitalisation_to_death_truncated_high)
    print(" 4 -- Saving CSV to output")
    date_tag = datetime.now().strftime("%Y-%m-%d-%Hh")
    allTogetherLow.to_csv("output/{}_low.csv".format(date_tag))
    allTogetherMid.to_csv("output/{}_mid.csv".format(date_tag))
    allTogetherHigh.to_csv("output/{}_high.csv".format(date_tag))
