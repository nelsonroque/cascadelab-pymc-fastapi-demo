from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import pymc as pm
import pandas as pd
import arviz as az
import math
import time

app = FastAPI()

# Define the input model for API
class ModelInput(BaseModel):
    num_participants: int = 300  # The number of participants
    days_per_person: int = 14  # The number of days for each person

# Step 1: Simulate data generation process
def generate_synthetic_data(P: int, days_per_person: int):
    # Set the random seed for reproducibility
    np.random.seed(42)

    N = P * days_per_person  # total number of datapoints

    # Create dayIndex: repeated measurements for each person
    dayIndex = np.tile(np.arange(0, days_per_person), P)

    # Create nrAssess: number of assessments for each day
    nrAssess = np.random.randint(1, 6, size=N)

    # Create personIndex: person identifiers
    personIndex = np.repeat(np.arange(P), days_per_person)

    # True parameters
    true_murC = 0.5
    true_mugC = 2.0
    true_muA = 1.0
    true_muSdE = 0.2

    true_rCSd = 0.1
    true_gCSd = 0.5
    true_aSd = 0.3
    true_sdESd = 0.1

    # Generate person-specific true parameters from the hyperpriors
    true_rC = np.abs(np.random.normal(true_murC, true_rCSd, P))
    true_gC = np.abs(np.random.normal(true_mugC, true_gCSd, P))
    true_a = np.abs(np.random.normal(true_muA, true_aSd, P))
    true_sdE = np.abs(np.random.normal(true_muSdE, true_sdESd, P))

    # Simulate RT data based on the true parameters
    RT = np.zeros(N)
    for n in range(N):
        person = personIndex[n]
        RT[n] = np.random.normal(
            true_a[person] + true_gC[person] * np.exp(-true_rC[person] * dayIndex[n]),
            true_sdE[person] / math.sqrt(nrAssess[n])
        )

    return RT, personIndex, dayIndex, nrAssess, P

# Step 2: PyMC Model
def run_pymc_model(RT, personIndex, dayIndex, nrAssess, P):
    start_time = time.time()

    with pm.Model() as model:

        # Hyperpriors
        murC = pm.HalfNormal("murC", sigma=1)
        mugC = pm.HalfNormal("mugC", sigma=1)
        muA = pm.HalfNormal("muA", sigma=1)
        muSdE = pm.HalfNormal("muSdE", sigma=1)

        rCSd = pm.HalfNormal("rCSd", sigma=1)
        gCSd = pm.HalfNormal("gCSd", sigma=1)
        aSd = pm.HalfNormal("aSd", sigma=1)
        sdESd = pm.HalfNormal("sdESd", sigma=1)

        # Person-specific parameters
        rC = pm.TruncatedNormal("rC", mu=murC, sigma=rCSd, lower=0, shape=P)
        gC = pm.TruncatedNormal("gC", mu=mugC, sigma=gCSd, lower=0, shape=P)
        a = pm.TruncatedNormal("a", mu=muA, sigma=aSd, lower=0, shape=P)
        sdE = pm.TruncatedNormal("sdE", mu=muSdE, sigma=sdESd, lower=0, shape=P)

        # Likelihood
        RT_obs = pm.Normal(
            "RT_obs",
            mu=a[personIndex] + gC[personIndex] * pm.math.exp(-rC[personIndex] * dayIndex),
            sigma=sdE[personIndex] / np.sqrt(nrAssess),
            observed=RT
        )

        # Sampling
        trace = pm.sample(draws=2000, chains=2, cores=2, return_inferencedata=True, random_seed=123)

    # End time for execution
    end_time = time.time()
    time_diff = (end_time - start_time) / 60
    print(f"Time taken: {time_diff:.2f} minutes")

    # Return the summary of the trace
    summary = az.summary(trace)
    return summary

# Step 3: Run the model in the background
def run_model_in_background(num_participants: int, days_per_person: int):
    # Step 1: Generate synthetic data
    RT, personIndex, dayIndex, nrAssess, P = generate_synthetic_data(num_participants, days_per_person)

    # Step 2: Run the PyMC model
    summary = run_pymc_model(RT, personIndex, dayIndex, nrAssess, P)

    # Print or save the result
    summary_df = summary.reset_index().rename(columns={'index': 'parameter'})
    summary_df.to_csv('resulttable.csv', index=False)
    print("Model run completed and results saved to resulttable.csv")

# FastAPI endpoint to trigger the model run
@app.post("/run-model/")
async def run_model(input_data: ModelInput, background_tasks: BackgroundTasks):
    # Add the model run to the background tasks
    background_tasks.add_task(run_model_in_background, input_data.num_participants, input_data.days_per_person)
    return {"message": "Model run started in the background"}
