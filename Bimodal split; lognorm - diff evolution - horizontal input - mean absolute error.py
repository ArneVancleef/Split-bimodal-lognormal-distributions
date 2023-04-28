"""
Created on Tue Mar 21 11:13:52 2023

@author: u0117522
"""

# import libraries
import numpy as np
#from numpy import log as log
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
#from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
#from bayes_opt import BayesianOptimization
#import pyswarms as ps
#from statsmodels.stats.weightstats import DescrStatsW
#from pyswarms.utils.plotters import plot_cost_history
#from scipy.optimize import minimize
#from scipy.optimize import minimize
from scipy.optimize import differential_evolution
#from skopt.plots import plot_gaussian_process
#from skopt import gp_minimize
#from skopt.plots import plot_convergence
#from sklearn.gaussian_process import GaussianProcessRegressor

# %% Read CSV file
df = pd.read_csv('C:/Users/u0117522/OneDrive - KU Leuven/Documents/Bimodal_distribution_examples.csv',";")
df = df.transpose()
df = df.reset_index()
df.columns = df.iloc[0]
df = df.drop(df.index[0])
df.head()
df = df.astype(float)
#error_list = []
column_list = []
Mean1_list = []
Mean2_list = []
Sigma1_list = []
Sigma2_list = []
W1_list = []
W2_list = []
mu1_list = []
mu2_list = []
R2_list = []
X = []
Y_exp = []

def fit(df,column,X,Y_exp):
    # %% Bimodal lognom distribution model and cost function
    def sim(S1,M1,S2,M2,W1,W2):
        #X = df["X"]
        mu1 = np.log(M1) - S1**2 / 2
        Y_1 =(1 / (X*S1*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(X)-mu1)**2)/(S1**2))
        mu2 = np.log(M2) - S2**2 / 2
        Y_2 =(1 / (X*S2*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(X)-mu2)**2)/(S2**2))
        Y_pred = W1*Y_1 + W2*Y_2
        return Y_pred
    
    def cost_func(params):
        S1 = params[0]
        M1 = params[1]
        S2 = params[2]
        M2 = params[3]
        W1 = params[4]
        W2 = params[5]
    
        Y_pred = np.array(sim(S1,M1,S2,M2,W1,W2),dtype=object)
        #Y_exp = df["Y"]
        error = mean_absolute_error(Y_pred,Y_exp)
        #error_list.append(error)
        return error
    
    # %%  Differential Evolution
    # This needs to be updated when 
    dimensions = 6
    iters = 2000000
    
    lower_b = [0.001, 5 ,0.1,50 ,10 ,100 ] #lower bound
    upper_b = [4, 100, 2.5, 3000 ,10000 ,500000 ] #upper bound
    
    bounds = [(lower_b[i], upper_b[i]) for i in range(dimensions)]
    
    result = differential_evolution(cost_func, bounds, maxiter=iters)
    
    coeff = result.x
    cost = result.fun
   
    # %% Validation of the obtained coefficients
    def Predict(X):
        S1 = coeff[0]
        M1 = coeff[1]
        S2 = coeff[2]
        M2 = coeff[3]
        W1 = coeff[4]
        W2 = coeff[5]
        mu1 = np.log(M1) - S1**2 / 2
        Y_1 =(1 / (X*S1*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(X)-mu1)**2)/(S1**2))
        mu2 = np.log(M2) - S2**2 / 2
        Y_2 =(1 / (X*S2*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(X)-mu2)**2)/(S2**2))
        Y_pred = W1*Y_1 + W2*Y_2
        return Y_pred
    
    def PredictY1(X):
        S1 = coeff[0]
        M1 = coeff[1]
        W1 = coeff[4]
        mu1 = np.log(M1) - S1**2 / 2
        return (1 / (X*S1*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(X)-mu1)**2)/(S1**2))*W1
    
    def PredictY2(X):
        S2 = coeff[2]
        M2 = coeff[3]
        W2 = coeff[5]
        mu2 = np.log(M2) - S2**2 / 2
        return (1 / (X*S2*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(X)-mu2)**2)/(S2**2))*W2
    
    Y_pred = Predict(X)
    Y_1pred =PredictY1(X)
    Y_2pred =PredictY2(X)
    
    S1 = round(coeff[0],2)
    M1 = round(coeff[1],2)
    S2 = round(coeff[2],2)
    M2 = round(coeff[3],2)
    W1 = round(coeff[4],2)
    W2 = round(coeff[5],2)
    
    
    sumY_exp =  round(np.sum(Y_exp))
    sumY_pred =  round(np.sum(Y_pred))
    Med1 = round(np.exp(np.log(M1) - S1**2 / 2))
    Med2 = round(np.exp(np.log(M2) - S2**2 / 2))
    Mode1 = round(np.exp((np.log(M1) - 0.5*S1**2)-S1**2))
    Mode2 = round(np.exp((np.log(M2) - 0.5*S2**2)-S2**2))
    mu1 = round(np.exp(np.log(M1) - S1**2 / 2))
    mu2 = round(np.exp(np.log(M2) - S2**2 / 2))
    R2 = round(r2_score(Y_pred,Y_exp),6)
    
    #print("The Weighted standard deviation of the distribution is: ",S)
    #print("The Weighted average of the distribution is: ",M)
    print(" ")
    print("The constants of the model are: ",coeff)
    print("Mean peak one =",M1)
    print("Mean peak two =",M2)
    print("Median peak one =",Med1)
    print("Median peak two =",Med2)
    print("Mode peak one =",Mode1)
    print("Mode peak two =",Mode2)
    print("R²-score model =",R2)
    column_list.append(column)
    Mean1_list.append(M1)
    Mean2_list.append(M2)
    Sigma1_list.append(S1)
    Sigma2_list.append(S2)
    W1_list.append(W1)
    W2_list.append(W2)
    mu1_list.append(mu1)
    mu2_list.append(mu2)
    R2_list.append(R2)

    
    # %% graphs
    # plt.figure(figsize = (15, 5))
    # plt.plot(range(1, 1 + len(optimizer.space.target)), optimizer.space.target, "-o")
    # plt.grid(True)
    # plt.xlabel("Iteration", fontsize = 14)
    # plt.ylabel("Black box function f(x)", fontsize = 14)
    # plt.xticks(fontsize = 14)
    # plt.yticks(fontsize = 14)
    # plt.show()
    
    """
    # Obtain cost history from optimizer instance
    cost_history = optimizer.cost_history
    # Plot!
    plot_cost_history(cost_history)
    plt.show()
    
    plt.plot(error_list,'.')
    plt.yscale('log')  # set X-axis to logarithmic scale
    plt.ylim(bottom = 1)
    plt.show
    """
    
    plt.figure()
    plt.title('Bimodal distribution {}'.format(column))
    plt.plot([0, np.max([Y_exp,Y_pred])], [0, np.max([Y_exp,Y_pred])], '--') 
    plt.plot(Y_exp,Y_pred, 'o')
    plt.xlabel('Experimental')
    plt.ylabel('Predicted')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Bimodal distribution {}'.format(column))
    plt.plot(X,Y_exp, 'black', label = 'Experimental')
    plt.plot(X,Y_pred, '--b', label = 'predicted')
    plt.xlabel('Particle size')
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Bimodal distribution {}'.format(column))
    plt.plot(X,Y_exp, 'black', label = 'Experimental')
    plt.plot(X,Y_pred, '--b', label = 'predicted')
    plt.plot(X,Y_1pred, '--g', label = 'predicted 1')
    plt.plot(X,Y_2pred, '--r', label = 'predicted 2')
    plt.plot([Mode2,Mode2],[0,4.2],'-y')
    plt.xlabel('Particle size')
    plt.ylabel('Percentage')
    plt.show()
    
    
    #X = df["X"]
    Xexpanded = np.linspace(1,10000,5000)
    #Y_exp = df["Y"]
    Y_predexpanded = Predict(Xexpanded)
    Y_pred1expanded = PredictY1(Xexpanded)
    Y_pred2expanded = PredictY2(Xexpanded)
    
    plt.figure()
    plt.title('Bimodal distribution {}'.format(column))
    plt.plot(X,Y_exp, 'black', label = 'Experimental')
    plt.plot(Xexpanded,Y_predexpanded, '--b', label = 'predicted')
    plt.plot(Xexpanded,Y_pred1expanded, '--g', label = 'predicted 1')
    plt.plot(Xexpanded,Y_pred2expanded, '--r', label = 'predicted 2')
    plt.xlabel('Particle size')
    plt.ylabel('Percentage')
    plt.xscale('log')  # set X-axis to logarithmic scale
    #plt.legend()
    plt.show()
    
    plt.figure()
    plt.title('Bimodal distribution {}'.format(column))
    plt.plot(X,Y_exp, 'black', label = 'Expected')
    plt.plot(Xexpanded,Y_predexpanded, '--b', label = 'predicted')
    plt.xlabel('Particle size')
    plt.ylabel('Percentage')
    plt.xscale('log')  # set X-axis to logarithmic scale
    plt.legend()
    plt.show()
    
    #%%
    """
    X = np.linspace(0.0001,10,5000)
    S1b = 0.2
    M1b = 2
    mu1b = np.log(M1b) - S1b**2 / 2
    S2b = 0.2
    M2b = 1
    mu2b = np.log(M2b) - S2b**2 / 2
    
    Y_1b =(1 / (X*S1b*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(X)-mu1b)**2)/(S1b**2))
    Y_2b =(1 / (X*S2b*np.sqrt(2*np.pi)))*np.exp(-0.5*((np.log(X)-mu2b)**2)/(S2b**2))
    Y_pred = 1*Y_1b + 1*Y_2b
    #mean = exp(mu + sigma^2 / 2)
    plt.plot(X,Y_1b, 'red')
    plt.plot(X,Y_2b, 'blue')
    plt.plot(X,Y_pred, 'orange')
    plt.show()
    """
    
i = 0
for col in df.iloc[:, 1:]:
    print(col)
    X = df["Particle size"]
    NewDf = np.column_stack((X, df[col])) # stack the two arrays horizontally
    NewDf = NewDf[~np.isnan(NewDf).any(axis=1)] # remove any row that contains NaN values
    X_new = NewDf[:, 0] # extract the X column
    X_new_series = pd.Series(X_new)
    Y_new = NewDf[:, 1] # extract the other columns
    Y_new_series = pd.Series(Y_new)
    fit(df, col, X_new_series, Y_new_series)
