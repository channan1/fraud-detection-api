import numpy as np
import pandas as pd
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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import wilcoxon
from scipy import stats
from sklearn.metrics import confusion_matrix
import math
import asyncio
import pickle

model_dict = {
    'LinearDiscriminantAnalysis': LDA(),
    'QuadraticDiscriminantAnalysis': QDA(),
    'GradientBoostingClassifier': GradientBoostingClassifier(n_estimators=50),
    'AdaBoostClassifier': AdaBoostClassifier(n_estimators=5, learning_rate=0.05),
    'RandomForestClassifier': RandomForestClassifier(),
    'LogisticRegression': LogisticRegression(),
    'KNeighborsClassifier': KNeighborsClassifier()
}

scoring_methods = {
    'accuracy':'accuracy',
    'precision':'precision',
    'recall':'recall',
    'f1':'f1'
}

param_grids = {
    'rf':{
        'n_estimators': np.arange(0,101,5)[1:].tolist(),
        'max_depth': [None],
        'bootstrap': [True,False]
    },
    'ada':{
        'n_estimators': np.arange(0,101,5)[1:].tolist(),
        'learning_rate': np.arange(0,1,0.05)[1:].tolist(),
        'algorithm': ['SAMME']
    }
}

def load_trained_models(models):
    trained_models = {}
    for model in models:
        with open(f"./data/{model}.pkl",'rb') as file:
            trained_models[model] = pickle.load(file)
    return trained_models

def load_data():
    data = pd.read_csv("./data/synthetic_fraud_dataset.csv")
    orig_data = data.copy()

    #Balance of 0's and 1's
    # print(data['Fraud_Label'].value_counts())

    y_col_name = 'Fraud_Label'
    cols_to_drop = ['Transaction_ID', 'User_ID', 'Timestamp', 'Risk_Score']
    cols_to_dummy = ['Transaction_Type', 'Device_Type', 'Location', 'Merchant_Category', 'Card_Type', 'Authentication_Method']
    test_size = 0.3
    data_for_prediction = data.drop(cols_to_drop, axis=1)
    data_with_dummies = pd.get_dummies(data_for_prediction, columns = cols_to_dummy, drop_first=True, dtype='int')
    data_with_dummies.rename(columns={'Location_New York':'Location_New_York', 'Transaction_Type_Bank Transfer':'Transaction_Type_Bank_Transfer', 'Authentication_Method_Password':'Authentication_Method_PW'}, inplace=True)
    data = data_with_dummies.copy()
    X_train_scaled_nhp, X_train_scaled_hp, X_val_scaled_hp, X_test_scaled_hp, X_test_scaled_nhp, y_test, y_val, y_train_nhp = pd.read_csv("./data/X_train_scaled_nhp.csv", index_col=0), pd.read_csv("./data/X_train_scaled_hp.csv", index_col=0), pd.read_csv("./data/X_val_scaled_hp.csv", index_col=0), pd.read_csv("./data/X_test_scaled_hp.csv", index_col=0), pd.read_csv("./data/X_test_scaled_nhp.csv", index_col=0), pd.read_csv("./data/y_test.csv", index_col=0), pd.read_csv("./data/y_val.csv", index_col=0), pd.read_csv("./data/y_train_nhp.csv", index_col=0)
    return data, orig_data, y_col_name, X_train_scaled_nhp, X_train_scaled_hp, X_val_scaled_hp, X_test_scaled_hp, X_test_scaled_nhp, y_test, y_val, y_train_nhp
    # return data, orig_data, y_col_name

def pre_train_models(models, y_col_name, data, orig_data):
    trained_models = {}
    for i in models:
        instance = FraudClassifiers(model_dict[i],{})
        X_train, X_val, X_test, X_train_nhp, y_train, y_test, y_val, y_train_nhp, X_train_nhp_columns = instance.split(y_col_name, data)
        X_train_scaled_hp, X_val_scaled_hp, X_test_scaled_hp, X_train_scaled_nhp, X_test_scaled_nhp = instance.scale(X_train, X_val, X_test, X_train_nhp)
        instance.train(X_train_scaled_nhp, y_train_nhp, valandtrainjoined=False, hyperparams = None)
        trained_models[i] = instance

        filename = f"{i}.pkl" #Final Portion that is excluded when training upon API startup
        with open(filename, 'wb') as file:
            pickle.dump(instance, file)

    X_train_nhp.to_csv("X_train_nhp.csv", index=True)
    X_test.to_csv("X_test.csv", index=True)
    X_val_scaled_hp.to_csv("X_val_scaled_hp.csv", index=True)
    pd.DataFrame(y_val, index=X_val.index).to_csv("y_val.csv", index=True)
    pd.DataFrame(y_train_nhp, index=X_train_nhp.index).to_csv("y_train_nhp.csv", index=True)
    pd.DataFrame(y_test, index=X_test.index).to_csv("y_test.csv", index=True)
    pd.DataFrame(y_train, index=X_train.index).to_csv("y_train.csv", index=True)
    X_train_scaled_nhp.to_csv("X_train_scaled_nhp.csv", index=True)
    X_test_scaled_nhp.to_csv("X_test_scaled_nhp.csv", index=True)
    X_test_scaled_hp.to_csv("X_test_scaled_hp.csv", index=True)
    X_train_scaled_hp.to_csv("X_train_scaled_hp.csv", index=True)

    return trained_models, X_train_scaled_hp, X_val_scaled_hp, X_test_scaled_hp, X_test_scaled_nhp, y_test, y_val, y_train_nhp

def descriptive_analytics(data):
    #===== Load Data =====
    data_for_eda = data.copy()
    data_for_prediction = data.copy()

    #=== Look at User IDs most associated with Fraud ===

    data_for_eda['NonFraud'] = 0
    data_for_eda.loc[data_for_eda['Fraud_Label'] == 0, 'NonFraud'] = 1
    #User ID Groupby
    uids = data_for_eda.groupby('User_ID').agg({'Fraud_Label':'sum', 'NonFraud':'sum'}).sort_values(by='Fraud_Label', ascending=False)
    uids['FraudPerc'] = uids['Fraud_Label'] / (uids['Fraud_Label'] + uids['NonFraud'])
    #City Groupby
    city = data_for_eda.groupby('Location').agg({'Fraud_Label':'sum', 'NonFraud':'sum'}).sort_values(by='Fraud_Label', ascending=False)
    city['FraudPerc'] = city['Fraud_Label'] / (city['Fraud_Label'] + city['NonFraud'])
    #Trantype Groupby
    tran = data_for_eda.groupby('Transaction_Type').agg({'Fraud_Label':'sum', 'NonFraud':'sum'}).sort_values(by='Fraud_Label', ascending=False)
    tran['FraudPerc'] = tran['Fraud_Label'] / (tran['Fraud_Label'] + tran['NonFraud'])
    #Merchant Category Groupby
    merch = data_for_eda.groupby('Merchant_Category').agg({'Fraud_Label':'sum', 'NonFraud':'sum'}).sort_values(by='Fraud_Label', ascending=False)
    merch['FraudPerc'] = merch['Fraud_Label'] / (merch['Fraud_Label'] + merch['NonFraud'])
    #Devicetype Groupby
    device = data_for_eda.groupby('Device_Type').agg({'Fraud_Label':'sum', 'NonFraud':'sum'}).sort_values(by='Fraud_Label', ascending=False)
    device['FraudPerc'] = device['Fraud_Label'] / (device['Fraud_Label'] + device['NonFraud'])
    #Card Type Groupby
    card = data_for_eda.groupby('Card_Type').agg({'Fraud_Label':'sum', 'NonFraud':'sum'}).sort_values(by='Fraud_Label', ascending=False)
    card['FraudPerc'] = card['Fraud_Label'] / (card['Fraud_Label'] + card['NonFraud'])
    #AuthMethod Groupby
    auth = data_for_eda.groupby('Authentication_Method').agg({'Fraud_Label':'sum', 'NonFraud':'sum'}).sort_values(by='Fraud_Label', ascending=False)
    auth['FraudPerc'] = auth['Fraud_Label'] / (auth['Fraud_Label'] + auth['NonFraud'])
    cat_ls = [city, tran, merch, device, card, auth]
    cat_ls_name = ['City', 'Transaction Type', 'Merchant Type', 'Device Type', 'Card Type' ,'Authentication Method']
    colors = ['#AEC6CF', '#FAC898', '#FF6961', '#77DD77', '#B39EB5', '#FADADD', '#C1E1C1', '#FFE5B4', '#E0BBE4', '#FFDAC1']
    colors2 = colors.copy()
    colors2.reverse()
    for i in range(len(cat_ls)):
        plt.barh(cat_ls[i].index, cat_ls[i]['FraudPerc'] * 100, color=colors)
        plt.title(f"Percentage of Fraud by {cat_ls_name[i]}")
        plt.xlabel('Percentage of Fraud')
        plt.ylabel(cat_ls_name[i])
        plt.xlim(left=25)
        plt.tight_layout()
        plt.savefig(f'FraudPerc by {cat_ls_name[i]}')
        plt.close()

    #User IDs
    # print(uids)
    # print(list(uids.iloc[:8].index), list(uids.iloc[:8, 2:].values.reshape(8,)))
    uids_x, uids_y = list(uids.iloc[:10, 0:1].index), list(uids.iloc[:10,0:1].values.reshape(10,))
    uids_x.reverse()
    uids_y.reverse()
    plt.barh(uids_x, uids_y, color=colors2)
    plt.title(f"Top 10 Instances of Fraud by User ID")
    plt.xlabel('Instances of Fraud')
    plt.ylabel("User ID")
    # plt.xlim(left=25)
    plt.tight_layout()
    plt.savefig(f'Fraud by User ID')
    plt.close()

    #====== Temporal Analysis =======
    data_temporal = data[['Timestamp', 'Fraud_Label']]
    data_temporal['Timestamp'] = pd.to_datetime(data_temporal['Timestamp'])
    data_temporal['NonFraud'] = 0
    data_temporal.loc[data_temporal['Fraud_Label'] == 0, 'NonFraud'] = 1

    data_temporal_dow = data_temporal.copy()
    data_temporal_dow['dow'] = data_temporal['Timestamp'].dt.weekday

    # data_temporal = data_temporal.groupby(data_temporal['Timestamp'].dt.date).sum()
    data_temporal_daily = data_temporal.set_index('Timestamp').groupby(pd.Grouper(freq='D')).sum()
    data_temporal_daily['FraudPerc'] = data_temporal_daily['Fraud_Label'] / (data_temporal_daily['Fraud_Label'] + data_temporal_daily['NonFraud'])
    data_temporal_monthly = data_temporal.set_index('Timestamp').groupby(pd.Grouper(freq='M')).sum()
    data_temporal_monthly['FraudPerc'] = data_temporal_monthly['Fraud_Label'] / (data_temporal_monthly['Fraud_Label'] + data_temporal_monthly['NonFraud'])

    # data_temporal['dow'] = data_temporal.index.dt.weekday
    data_temporal_dow = data_temporal_dow.drop('Timestamp', axis=1)
    data_temporal_dow['NonFraud'] = 0
    data_temporal_dow.loc[data_temporal_dow['Fraud_Label'] == 0, 'NonFraud'] = 1
    # print(data_temporal_dow)
    dow_grouped = data_temporal_dow.groupby('dow').sum()
    dow_grouped['FraudPerc'] = dow_grouped['Fraud_Label'] / (dow_grouped['Fraud_Label'] + dow_grouped['NonFraud'])
    # print(dow_grouped)
    dows = ['Mon', 'Tue', 'Wed', 'Thur', 'Fri', 'Sat', 'Sun']

    plt.plot(dows, dow_grouped['FraudPerc'])
    plt.title('Percentage of Fraud Instances by Day of Week')
    plt.xlabel("Day of Week")
    plt.ylabel('Fraud Percentage')
    plt.savefig('Perc of Fraud by DOW')
    plt.close()

    plt.plot(data_temporal_daily.index, data_temporal_daily['FraudPerc'])
    plt.xlabel('Date')
    plt.ylabel('Percentage of Fraud')
    plt.title('Percentage of Fraud Instances by Date')
    plt.savefig("Percentage of Fraud by Date")
    # plt.show()
    plt.close()

    plt.plot(data_temporal_monthly.index, data_temporal_monthly['FraudPerc'])
    plt.xlabel('Date')
    plt.ylabel('Percentage of Fraud')
    plt.title('Percenatage of Fraud Instances by Month')
    plt.savefig("Percentage of Fraud by Month")
    # plt.show()
    plt.close()

    #====== EDA with Binary and Continuous Variables ========
    keep = ['Transaction_Amount', 'Account_Balance', 'IP_Address_Flag', 'Previous_Fraudulent_Activity', 'Daily_Transaction_Count', 'Avg_Transaction_Amount_7d', 'Failed_Transaction_Count_7d', 'Card_Age', 'Transaction_Distance', 'Is_Weekend']
    data_cont_eda = data[keep]
    data_cont_eda['Fraud_Label'] = data['Fraud_Label']

    corr_matrix = data_cont_eda.corr()
    plt.figure(figsize=(16,12))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm_r", vmin=-1, vmax=1, linewidth=0.5, annot_kws={'fontsize':16})
    plt.title("Correlation Matrix - Binary and Cont. Variables")
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=25)
    plt.tight_layout()
    # plt.show()
    plt.savefig("corrplot")
    plt.close()

    #======== Conduct Accuracy Test with Risk_Score Field ========
    #Attempt to outperform it
    data_for_eda['risk_solid_score'] = (data_for_eda['Risk_Score']>=0.5).astype(int)
    existing_score = error(data_for_eda['risk_solid_score'], data_for_eda['Fraud_Label'])
    print("CURRENT STATE RISK SCORE ERROR")
    print(existing_score)
    print()

    #Balance of 0's and 1's
    print(data_for_prediction['Fraud_Label'].value_counts())

def error(y_pred, y_actual):
    if y_pred.shape[0] != y_actual.shape[0]:
        print('Vectors do not match')
        return 0
    doesnt_match = (y_pred != y_actual).sum()
    return doesnt_match / y_pred.shape[0]

class FraudClassifiers():
    def __init__(self, model, hyperparameters: dict):
        self.model = model
        self.hyperparameters = hyperparameters
        self.model_name = type(self.model).__name__
    
    def split(self, y_col_name: str, formatted_df: pd.DataFrame, test_size: float=0.3, rand_state: int = 12345678):
        y = formatted_df[y_col_name]
        X = formatted_df.drop([y_col_name], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rand_state)
        X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=rand_state)
        X_train_nhp, y_train_nhp = pd.concat([X_train, X_val]), pd.concat([y_train, y_val])

        #Save Unscaled Train, Val and Test splits
        # self.X_train_unscaled_hp, self.X_val_unscaled_hp, self.X_test_unscaled_hp = X_train, X_val, X_test

        #Save Unscaled Train Set for Non-Hyperparameter Models
        # self.X_train_unscaled_nhp = X_train_nhp

        #Save All Y-Splits Which remain Unscaled
        # self.y_train_hp, self.y_test_hp, self.y_val_hp = y_train, y_test.to_numpy(), y_val.to_numpy()

        #Save All Y-Splits for Non-Hyperparam Models which remain Unscaled
        # self.y_train_nhp = y_train_nhp.to_numpy()

        # X_train_nhp.to_csv("X_train_nhp.csv", index=True)
        # X_test.to_csv("X_test.csv", index=True)
        # y_train_nhp.to_csv("y_train_nhp.csv", index=True)
        # y_test.to_csv("y_test.csv", index=True)

        # self.columns = X_train_nhp.columns

        return X_train, X_val, X_test, X_train_nhp, y_train, y_test.to_numpy(), y_val.to_numpy(), y_train_nhp.to_numpy(), X_train_nhp.columns


    def scale(self, X_train_unscaled_hp, X_val_unscaled_hp, X_test_unscaled_hp, X_train_unscaled_nhp):
        #Handle Data for Models that need Hyperparam tuning first (Train Val Test)
        scaler_hp = StandardScaler()

        X_train_scaled_hp = pd.DataFrame(scaler_hp.fit_transform(X_train_unscaled_hp),columns=X_train_unscaled_hp.columns,index=X_train_unscaled_hp.index) #self.X_train_unscaled_hp
        X_val_scaled_hp, X_test_scaled_hp = pd.DataFrame(scaler_hp.transform(X_val_unscaled_hp), columns=X_val_unscaled_hp.columns, index=X_val_unscaled_hp.index), pd.DataFrame(scaler_hp.transform(X_test_unscaled_hp), columns=X_test_unscaled_hp.columns, index=X_test_unscaled_hp.index) #self.X_val_unscaled_hp self.X_test_unscaled_hp

        #Handle Data for Models that don't require Hyperparam tuning second (Train Test)
        scaler_nhp = StandardScaler()
        X_train_scaled_nhp = pd.DataFrame(scaler_nhp.fit_transform(X_train_unscaled_nhp), columns=X_train_unscaled_nhp.columns, index=X_train_unscaled_nhp.index) #self.X_train_unscaled_nhp
        X_test_scaled_nhp = pd.DataFrame(scaler_nhp.transform(X_test_unscaled_hp), columns=X_test_unscaled_hp.columns, index=X_test_unscaled_hp.index) #In this case, X_test_hp and X_test_nhp are always the same thing, but need to be scaled again due to merge of train and val set for non-hp methods. #self.X_test_unscaled_hp
        # pd.DataFrame(self.X_train_scaled_nhp, columns=self.columns).to_csv("X_train_scaled_nhp.csv", index=True)
        # pd.DataFrame(self.X_test_scaled_nhp, columns=self.columns).to_csv("X_test_scaled_nhp.csv", index=True)

        return X_train_scaled_hp, X_val_scaled_hp, X_test_scaled_hp, X_train_scaled_nhp, X_test_scaled_nhp


    def error(self, y_pred, y_actual):
        if y_pred.shape[0] != y_actual.shape[0]:
            print('Vectors do not match')
            return 0
        doesnt_match = (y_pred != y_actual).sum()
        return doesnt_match / y_pred.shape[0]
    
    def train(self, X_train_scaled, y_train, valandtrainjoined=False, hyperparams = None):
        assert self.model_name in ['RandomForestClassifier', 'AdaBoostClassifier','LogisticRegression', 'LinearDiscriminantAnalysis', 'KNeighborsClassifier', 'QuadraticDiscriminantAnalysis', 'GradientBoostingClassifier']
        if hyperparams is not None:
            #Set hyperparams into chosen model here
            self.model.set_params(**hyperparams)
        # if self.model_name in ['LinearDiscriminantAnalysis', 'LogisticRegression', 'QuadraticDiscriminantAnalysis']: #Used originally, but may be obsolete.
        if valandtrainjoined:
            self.model.fit(X_train_scaled, y_train) #self x_train_scaled_nhp, y_train_nhp
        else:
            self.model.fit(X_train_scaled, y_train) #self X_train_scaled_hp, y_train_hp
    
    def query(self, X_test_scaled, y_test, val=False, single=False, single_obs=None, proba: float = 0.5):
        if single:
            assert single_obs is not None
            if isinstance(single_obs, np.ndarray):
                single_obs = single_obs.tolist()
            prediction_proba = self.model.predict_proba(single_obs)[:,1]
            prediction = np.where(prediction_proba >= proba, 1, 0)
            return prediction[0].item(), f"{float(prediction_proba[0].item()):.2%}"
        if not val:
            if self.model_name in ['LinearDiscriminantAnalysis', 'LogisticRegression', 'QuadraticDiscriminantAnalysis']:
                # predictions = self.model.predict(self.X_test_scaled_nhp) Saved this in case of future desire to return strict 0.5 threshold, otherwise the np.where line results in more dynamic ability to set threshold
                predictions_proba = self.model.predict_proba(X_test_scaled)[:,1] #self.X_test_scaled_nhp
                predictions = np.where(predictions_proba >= proba, 1, 0)
            else:
                # predictions = self.model.predict(self.X_test_scaled_hp) Saved this in case of future desire to return strict 0.5 threshold, otherwise the np.where line results in more dynamic ability to set threshold
                predictions_proba = self.model.predict_proba(X_test_scaled)[:,1] #self.X_test_scaled_nhp
                predictions = np.where(predictions_proba >= proba, 1, 0)
            return predictions, predictions_proba, y_test #self.y_test_hp
        else:
            assert self.model_name in ['RandomForestClassifier', 'AdaBoostClassifier','LogisticRegression', 'LinearDiscriminantAnalysis', 'KNeighborsClassifier', 'QuadraticDiscriminantAnalysis', 'GradientBoostingClassifier']
            # predictions = self.model.predict(self.X_val_scaled_hp) Saved this in case of future desire to return strict 0.5 threshold, otherwise the np.where line results in more dynamic ability to set threshold
            predictions_proba = self.model.predict_proba(X_test_scaled)[:,1] #self.X_val_scaled_hp
            predictions = np.where(predictions_proba >= proba, 1, 0)
            
            return predictions, predictions_proba, y_test #self.y_val_hp

    def calculate_metrics(self, predictions: np.ndarray, prediction_proba: np.ndarray, actual: np.ndarray) -> dict:
        accuracy = accuracy_score(actual, predictions)
        precision = precision_score(actual, predictions)
        recall = recall_score(actual, predictions)
        f1 = f1_score(actual, predictions)
        conf_mat = confusion_matrix(actual, predictions)
        roc_auc = roc_auc_score(actual, prediction_proba)
        return {
            'accuracy': str(f"{accuracy:.4f}"),
            'precision': str(f"{precision:.4f}"),
            'recall': str(f"{recall:.4f}"),
            'f1': str(f"{f1:.4f}"),
            'roc_auc': str(f"{roc_auc:.4f}"),
            'conf_mat': conf_mat.flatten().tolist()
        }
    
    def find_optimal_hyperparams_MCCV(self, MCCV_iters: int = 100, grid_search: dict ={}):
        ''' grid search will include comprehensive list of all values to try '''
        assert self.model_name not in ['LinearDiscriminantAnalysis', 'LogisticRegression', 'QuadraticDiscriminantAnalysis']
        assert len(grid_search)>0
        if len(grid_search) > 1:
            #Assign a combination to be taken
            pass
        elif len(grid_search) == 1:
            grid = list(grid_search.values())[0]
            avg_err = []
            for item in grid:
                inter_MCCV_obs = []
                for i in range(MCCV_iters):
                    self.split(y_col_name, 0.3, rand_state=12345678+i)
                    self.scale()
                    self.train(hyperparams={'n_estimators':item})
                    _, err = self.query(val=True)
                    inter_MCCV_obs.append(err)
                avg_err.append(np.mean(inter_MCCV_obs))
            best_error = np.min(avg_err)
            best_estimator = grid.index(np.min(avg_err))
            return best_error, best_estimator
    
    def find_optimal_hyperparams_KFoldGrid(self, y_col_name: str, ks: int =5, param_grid: dict ={}, scoring_method: str ='accuracy', verbose: int =3, rand_state: int =12345678):
        self.split(y_col_name, 0.3, rand_state=rand_state)
        self.scale()
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=ks, scoring=scoring_method, n_jobs=-1, verbose=verbose)
        grid_search.fit(self.X_train_scaled_hp, self.y_train_hp)
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        return best_estimator, best_params, best_score