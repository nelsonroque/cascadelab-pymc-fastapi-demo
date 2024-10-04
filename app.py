from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import pymc as pm
import numpy as np
from io import StringIO
import time

app = FastAPI(title="M2C2 Bayesian models in the cloud!")

# GET ALL THIS
# posterior probabilities (form credible intervals/HDI from the distribution of paramter esitmaes) = measure of certainty
# get priors even if defaults (necessary for papers)
# get the trace plot - 
# probably both the trace plot and the posterior plot
# rhat values
# divergences
# elapsed time
#how many samples, how many chains, how many iterations
# optioon to change iterations
# 4 chains, 2000 iterations; override for mlm
# effective sample size for the tail and the body
# power in bayseins is weird
 
@app.post("/run-model/sample/100/chains/19/iterations/1000")
async def run_model(file: UploadFile = File(...)):
    
    # start a timer
    start = time.time()
    
    # Check if uploaded file is CSV
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    # Read CSV file
    contents = await file.read()
    csv_data = StringIO(contents.decode("utf-8"))
    data = pd.read_csv(csv_data)

    # Ensure the CSV has 'x' and 'y' columns
    if 'x' not in data.columns or 'y' not in data.columns:
        raise HTTPException(status_code=400, detail="CSV must have 'x' and 'y' columns for this model.")

    x = data['x'].values
    y = data['y'].values

    # PyMC Bayesian Linear Regression model
    with pm.Model() as model:
        # Priors for unknown model parameters
        alpha = pm.Normal("alpha", mu=0, sigma=10)
        beta = pm.Normal("beta", mu=0, sigma=10)
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Expected value of outcome
        mu = alpha + beta * x

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Inference
        trace = pm.sample(1000, return_inferencedata=True)

    summary = pm.summary(trace)
    
    elapsed_time = time.time() - start
    
    output_data = summary.to_dict()
    output_data['elapsed_time'] = elapsed_time
    
    # extract info on divergences
    divergences = trace.sample_stats["diverging"]
    print(divergences) # count booleans in list of lists

    return output_data

@app.post("/run-model/mlm")
async def run_model(file: UploadFile = File(...)):
    
    # possible change the priors 
    # scale the priors to the data
    # truncated normal distribution (for Rt, gamma, etc)
    # change the link function to be normal, categorical
    # trunbcating outcome
       
    # start a timer
    start = time.time()
    
    # Check if uploaded file is CSV
    if file.content_type != 'text/csv':
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a CSV file.")
    
    # Read CSV file
    contents = await file.read()
    csv_data = StringIO(contents.decode("utf-8"))
    data = pd.read_csv(csv_data)

    # Ensure the CSV has 'x', 'y', and 'group' columns
    if 'x' not in data.columns or 'y' not in data.columns or 'group' not in data.columns:
        raise HTTPException(status_code=400, detail="CSV must have 'x', 'y', and 'group' columns for this multilevel model.")

    x = data['x'].values
    y = data['y'].values
    group = data['group'].values

    # Number of unique groups
    n_groups = len(np.unique(group))
    
    # PyMC Multilevel Bayesian Linear Regression model
    with pm.Model() as model:
        # Hyperpriors for group-level distributions
        mu_alpha = pm.Normal("mu_alpha", mu=0, sigma=10)
        mu_beta = pm.Normal("mu_beta", mu=0, sigma=10)
        sigma_alpha = pm.HalfNormal("sigma_alpha", sigma=1)
        sigma_beta = pm.HalfNormal("sigma_beta", sigma=1)
        
        # Group-specific intercepts and slopes
        alpha = pm.Normal("alpha", mu=mu_alpha, sigma=sigma_alpha, shape=n_groups)
        beta = pm.Normal("beta", mu=mu_beta, sigma=sigma_beta, shape=n_groups)

        # Priors for the error term
        sigma = pm.HalfNormal("sigma", sigma=1)

        # Group-specific intercepts and slopes for each observation
        alpha_group = alpha[group]
        beta_group = beta[group]

        # Expected value of outcome
        mu = alpha_group + beta_group * x

        # Likelihood (sampling distribution) of observations
        y_obs = pm.Normal("y_obs", mu=mu, sigma=sigma, observed=y)

        # Inference
        trace = pm.sample(1000, return_inferencedata=True)

    # Summarize the results
    summary = pm.summary(trace)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start
    
    # Convert the summary to a dictionary
    output_data = summary.to_dict()
    output_data['elapsed_time'] = elapsed_time
    
    # Extract info on divergences
    divergences = trace.sample_stats["diverging"].sum().item()  # count the number of divergences
    output_data['divergences'] = divergences
    
    # trace plot data
    # trace_data = pm.trace_to_dataframe(trace)
    # output_data['trace_data'] = trace_data.to_dict()
    
    # rhat values
    # want them between 1.0 and 1.01 #Mthe lower the better
    rhat_values = pm.rhat(trace)
    output_data['rhat_values'] = rhat_values.to_dict()
    
    # posterior distribution data
    # posterior_data = pm.posterior_to_xarray(trace)
    # output_data['posterior_data'] = posterior_data.to_dict()

    return output_data

# pymc, vs pystan vs tensorflow probability vs pybrms, vs bambi (syntax, lme4)