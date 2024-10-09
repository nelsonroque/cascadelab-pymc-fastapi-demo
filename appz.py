from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import pymc as pm
import pandas as pd
from typing import List
import arviz as az


app = FastAPI()

# Define the model for the request
class ModelInput(BaseModel):
    num_samples: int = 100  # The number of samples to generate

# Function to generate random data (simulating CSV loading)
def generate_random_data(num_samples: int):
    np.random.seed(42)
    data = {
        "x": np.random.normal(loc=0, scale=1, size=num_samples),
        "y": np.random.normal(loc=0, scale=1, size=num_samples),
    }
    return pd.DataFrame(data)
def run_pymc_model(data: pd.DataFrame):
    with pm.Model() as model:
        # Define priors
        alpha = pm.Normal('alpha', mu=0, sigma=10)
        beta = pm.Normal('beta', mu=0, sigma=10)
        sigma = pm.HalfNormal('sigma', sigma=1)

        # Define likelihood
        x = data['x'].values
        y = data['y'].values
        mu = alpha + beta * x
        likelihood = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

        # Sample from the posterior
        trace = pm.sample(1000, return_inferencedata=True)  # Ensure return_inferencedata=True
        
        # Summarize and return the InferenceData
        summary = az.summary(trace)  # az is short for ArviZ, which handles InferenceData objects
        print(summary)
        return summary

# Define the background task
def run_model_in_background(num_samples: int):
    data = generate_random_data(num_samples)
    result = run_pymc_model(data)
    # You can save the result to a file or a database
    print("Model run completed")

# Endpoint to trigger the model run
@app.post("/run-model/")
async def run_model(input_data: ModelInput, background_tasks: BackgroundTasks):
    # Add the model run to the background tasks
    background_tasks.add_task(run_model_in_background, input_data.num_samples)
    return {"message": "Model run started in the background"}
