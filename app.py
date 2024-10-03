from fastapi import FastAPI, UploadFile, File, HTTPException
import pandas as pd
import pymc as pm
import numpy as np
from io import StringIO
import time

app = FastAPI(title="M2C2 Bayesian models in the cloud!")

@app.post("/run-model/")
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

    return output_data

