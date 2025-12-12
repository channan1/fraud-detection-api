import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from scipy import stats
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.preprocessing import StandardScaler
import sys
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statistics import mode
import itertools
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wilcoxon
from scipy import stats
from sklearn.metrics import confusion_matrix
from ml_functions import FraudClassifiers, load_data, model_dict, param_grids, pre_train_models, load_trained_models
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import asyncio
import time
from typing import List, Optional
import pickle
from pydantic import BaseModel

np.set_printoptions(threshold=np.inf)

data, orig_data, y_col_name, X_train_scaled_nhp, X_train_scaled_hp, X_val_scaled_hp, X_test_scaled_hp, X_test_scaled_nhp, y_test, y_val, y_train_nhp = load_data()

models = list(model_dict.keys())

trained_models = load_trained_models(models)

app = FastAPI()

app.mount("/static", StaticFiles(directory='static'), name="static")
templates = Jinja2Templates(directory='templates')

@app.get('/')
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "models": models})

@app.get('/query')
def query_page(request: Request):
     return templates.TemplateResponse("query.html", {"request": request, "models": models})

@app.get('/performance')
def query_page(request: Request):
     return templates.TemplateResponse("performance.html", {"request": request, "models": models})

@app.get('/descriptors')
def query_page(request: Request):
     return templates.TemplateResponse("descriptors.html", {"request": request, "models": models})

@app.get('/static')
def get_static():
     return FileResponse('static')

@app.get('/health')
async def health_check():
     return {'status':'ok'}

class ModelEval(BaseModel):
     model: str
     proba: float

@app.post('/metrics/')
def predict(metrics: ModelEval):
        if metrics.model not in models:
             raise HTTPException(status_code=400, detail="Model not found")
        if metrics.proba < 0 or metrics.proba > 1:
             raise HTTPException(status_code=400, detail="Probability Threshold must be between 0 and 1.")
        instance = trained_models[metrics.model]
        predictions, predictions_proba, actual = instance.query(X_test_scaled_nhp, y_test, proba=metrics.proba)
        metrix = instance.calculate_metrics(predictions, predictions_proba, actual)
        return metrix

class SinglePredict(BaseModel):
     model: str
     index: int
     proba: float

@app.post('/obs_predict/')
def single_predict(obs_predict: SinglePredict):
    if obs_predict.model not in models:
             raise HTTPException(status_code=400, detail="Model not found")
    if obs_predict.index >= 7500 or obs_predict.index < 0:
         raise HTTPException(status_code=400, detail="Index value must be greater than or equal to 0 and less than 7500.")
    if obs_predict.proba < 0 or obs_predict.proba > 1:
         raise HTTPException(status_code=400, detail="Probability Threshold must be between 0 and 1.")
    instance = trained_models[obs_predict.model]
    ind_of_test = X_test_scaled_hp.index.tolist()[obs_predict.index]
    actual = int(data.loc[ind_of_test]['Fraud_Label'])
    obs = [X_test_scaled_hp.loc[ind_of_test].tolist()]

    if len(obs[0]) != 29: raise HTTPException(status_code=400, detail='Model requires 29 inputs in order to make predictions')
    else: prediction, prediction_proba = instance.query(X_test_scaled_nhp, y_test, single=True, single_obs=obs, proba=obs_predict.proba)
    obs_orig = orig_data.loc[ind_of_test].tolist()
    obs_orig = [str(element) for element in obs_orig]
    orig_col_names = orig_data.columns.tolist()
    actual_proba = f"{float(obs_orig[orig_col_names.index('Risk_Score')])*100:.2f}/100"
    obs_orig.pop(-3)
    orig_col_names.pop(-3)
    return {
         "prediction":prediction,
         'prediction_proba': prediction_proba,
         "actual":actual,
         "actual_proba": actual_proba,
         "headers": orig_col_names,
         "observation": obs_orig
         }

class CompPredict(BaseModel):
     index: int
     proba: float

@app.post('/comparative_obs_predict/')
def comp_predict(comp_obs_predict: CompPredict):
    if comp_obs_predict.index >= 7500 or comp_obs_predict.index < 0:
         raise HTTPException(status_code=400, detail="Index value must be greater than or equal to 0 and less than 7500.")
    if comp_obs_predict.proba < 0 or comp_obs_predict.proba > 1:
         raise HTTPException(status_code=400, detail="Probability Threshold must be between 0 and 1.")
    ind_of_test = X_test_scaled_hp.index.tolist()[comp_obs_predict.index]
    actual = int(data.loc[ind_of_test]['Fraud_Label'])
    obs = [X_test_scaled_hp.loc[ind_of_test].tolist()]
    if len(obs[0]) != 29: raise HTTPException(status_code=400, detail='Model requires 29 inputs in order to make predictions')
    obs_orig = orig_data.loc[ind_of_test].tolist()
    obs_orig = [str(element) for element in obs_orig]
    orig_col_names = orig_data.columns.tolist()
    actual_proba = f"{float(obs_orig[orig_col_names.index('Risk_Score')])*100:.2f}%"
    obs_orig.pop(-3)
    orig_col_names.pop(-3)
    predictions, predictions_proba = {}, {}
    for model in models:
         instance = trained_models[model]
         predictions[model], predictions_proba[model] = instance.query(X_test_scaled_nhp, y_test, single=True, single_obs=obs, proba=comp_obs_predict.proba)
    return {
         "prediction": predictions,
         'prediction_proba': predictions_proba,
         "actual":actual,
         "actual_proba": actual_proba,
         "headers": orig_col_names,
         "observation": obs_orig
     }