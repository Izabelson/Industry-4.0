import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
import re
import pickle


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error 



# Dataframe modification to be more readable



def mod_df(df):
    df.columns = ['Wavelenght [nm]', 'Amplitude']
    df['Wavelenght [nm]'] = df['Wavelenght [nm]'] * 1000000000
    return
#A function generatig of the dataframes with Sensor ID extracted in 'Sensor_ID' column from the file path.

# define file path and name of your dataframe 
# example of path: air_path = sorted(glob.glob(data_folder + '*0[0-8]_air.txt'))


def generate_dataframes(file_path, df_name):

    files = []

    for filename in file_path:
        df = pd.read_csv(filename, header=None, index_col=None)
        mod_df(df)
        df['Data_path']=filename
        df = df.astype({'Data_path':'string'})
        df['SensorID'] = df['Data_path'].str.extract(pat='(sensor[0-9][0-9])', expand=True)
        df = df.drop(columns=['Data_path'])
        df = df[['SensorID'] + list(df.columns[:-1])]
        files.append(df)

    df_name = pd.concat(files, axis=0, ignore_index=True)
    return df_name


'''

Function allows to plot Amplitude in function of Wavelenght in nm of each Sensor splitted by SensorID

'''


def spectrum_plot(data):

    fig, ax = plt.subplots(figsize=(10,6))
    sensors = list(data.SensorID.unique())
    for s in sensors:
        df = data.loc[data['SensorID'] == s]
        ax.plot(df['Wavelenght [nm]'], df['Amplitude'])
    ax.set_xlabel("Wavelenght, nm")
    ax.set_ylabel("P/Pref [dB]")
    return

plt.show()
import pickle


def poly_reg_model(X, y, poly_degree, filename):
    
    '''
    
    Function that fits polynominal regression model of X and y data with polynomial degree = poly_degree and predicts data.
    
    Function prints characteristic of polynomial regression (coefficients) and returns scores of the model (R2, MAE MSE, RMSE)
    
    Function saves the model to a file.
    
    Determine training X and training y and polynominal degree, and filename, which is the name of the file where the model will be saved.
    
    X : X training data; variable
    y: y training data, variable
    poly_degree: int, polynomial degree
    filename: string, name of the file where the model will be saved
    
    '''

    # transformation X
    poly = PolynomialFeatures(degree=poly_degree) 
    X_poly = poly.fit_transform(X)

    # model fitting
    model = LinearRegression()
    model_fit = model.fit(X_poly, y)

    print(f'Polynominal regression model with degree {poly_degree} function is characterized with COEFFICIENTS: {model.coef_}')

    #save model to file
    
    filename = filename + '.pkl'
    # serialize (save) the object, in this case "model"
    pickle.dump(model, open(filename, 'wb'))
    
    return model_fit, model.score(X_poly, y)
#prediction with the saved model

def make_predictions(X, y, filename):
# determine X which is formatted to polynominal function

# de-serialize (load) the object
    filename = filename + '.pkl'
    imported_model = pickle.load(open(filename, 'rb'))

# use the imported model (example)
    y_pred = imported_model.predict(X)

# print scatterplot of true values to predicted
    plt.scatter(y, y_pred)


    # parameters
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    rmse= np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print('Parameters of model:')
    print(f'MAE: {mae}, MSE: {mse}, RMSE: {rmse}, R2: {r2}')
    
    model = input('Enter the name of the model: ')
    dataset = input('Enter the name of the dataset: ')
    
    metrics_row = pd.DataFrame({'Model':model, 'Dataset':dataset, 'R2':r2, 'MAE':mae, 'MSE':mse, 'RMSE':rmse}, index=[0])
    
    return mae, mse, rmse, r2, metrics_row
 #Function to fit different models
from sklearn.pipeline import Pipeline

def fit_polynomials(X, y, from_=1, to_= 10, step=1):
  '''
  This function takes the explanatory (X) and response variables (y) and runs the data through a pipeline that fits Linear Regressions
  of different degrees (values specified by the user) and plot the results.
  Inputs:
  * from: int = initial degree for polynomial fit
  * to: int = final degree for polynomial fit
  * step:int = step to increase
  * X = explanatory variables
  * y = target variable

  Returns:
  score
  '''

  # Store scores and predictions
  scores = []

  # Loop between the specified values
  for n in range(from_, to_+1, step):
    # Steps
    steps = [
        ('Polynomial', PolynomialFeatures(degree=n)),
        ('model', LinearRegression())  ]
    
    # Pipeline fit
    fit_poly = Pipeline(steps).fit(X,y)
    # Predict
    poly_pred = fit_poly.predict(X)
    
    # y : y_predict scatter plot
    fig, ax = plt.subplots(figsize = (4, 4))
    ax.set_title("Prediction scatterplot for polynomial degree = " + str(n))
    ax.scatter(y, poly_pred)
    plt.show()
    
    # Evaluate
    model_score = fit_poly.score(X,y)
    model_mae = mean_absolute_error(y, poly_pred)
    model_mse = mean_squared_error(y, poly_pred)
    scores.append((n, model_score, model_mae, model_mse))

    return scores


import pickle


def dec_tree_model(X, y, max_depth, filename):
    
    '''
    
    Function that fits decision tree regression model of X and y data with max_depth = max_depth.
    
    Function prints characteristic of regression (score)
    
    Function saves the model to a file.
    
    Determine training X and training y and max_depth, and filename, which is the name of the file where the model will be saved.
    
    X : X training data; variable
    y: y training data, variable
    max_depth: int, max_depth of the tree
    filename: string, name of the file where the model will be saved
    
    '''

    model_dt = DecisionTreeRegressor(random_state=111, max_depth=max_depth)
    model_dt.fit(X, y)


    # score of the model
    score = model_dt.score(X, y)
    
    # plotting of the tree

    fig = plt.figure(figsize=(50,40), dpi=100)
    tree_fig = tree.plot_tree(model_dt,feature_names=X.columns, filled=True)

    print(f'Decission tree regression model with max_depth={max_depth} is characterized with SCORE: {score}')

    #save model to file
    
    filename = filename + '.pkl'
    # serialize (save) the object, in this case "model"
    pickle.dump(model_dt, open(filename, 'wb'))
    
    return model_dt, score