import pandas as pd
import numpy as np
from sklearn.metrics import r2_score

def prepare_dataset(dataset):

    # Taking Care of Missing Data
    dataset = dataset.dropna()

    # Splitting the Dataset into Target and Input Variables
    X = dataset.drop(['median_house_value', ], axis=1).values
    Y = dataset['median_house_value'].values.reshape(-1, 1)

    # Scaling the Data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X[:, :8] = scaler.fit_transform(X[:, :8])
    Y = scaler.fit_transform(Y)

    # Encoding Categorical Variables
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [8])], remainder='passthrough')
    X = ct.fit_transform(X)

    # Splitting the Dataset into Train and Test Sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

    return X_train, X_test, Y_train, Y_test

def linear_regression(X_train, X_test, Y_train, Y_test):
    from sklearn.linear_model import LinearRegression

    # Initializing the Model
    lr = LinearRegression()

    # Fitting the Training Data
    lr.fit(X_train, Y_train)

    # Predicting on Test Set
    Y_pred = lr.predict(X_test)

    # Evaluating the Model
    lr_ac = r2_score(Y_test, Y_pred)
    print(f'\nLINEAR REGRESSION \nAccuracy: {lr_ac}')

    return lr_ac

def svr(X_train, X_test, Y_train, Y_test):
    from sklearn.svm import SVR

    # Initializing the Model
    svr = SVR(kernel='rbf')

    # Fitting the Training Data
    svr.fit(X_train, Y_train.ravel())

    # Predicting on Test Set
    Y_pred = svr.predict(X_test)

    # Evaluating the Model
    svr_ac = r2_score(Y_test, Y_pred)
    print(f'\n\nSUPPORT VECTOR REGRESSOR \nAccuracy: {svr_ac}')

    return svr_ac

def decision_tree(X_train, X_test, Y_train, Y_test):
    from sklearn.tree import DecisionTreeRegressor

    # Initializing the Model
    dtc = DecisionTreeRegressor()

    # Fitting the Training Data
    dtc.fit(X_train, Y_train)

    # Predicting on Test Set
    Y_pred = dtc.predict(X_test)

    # Evaluating the Model
    dtr_ac = r2_score(Y_test, Y_pred)
    print(f'\n\nDECISION TREE REGRESSOR\nAccuracy: {dtr_ac}')

    return dtr_ac

def random_forest(X_train, X_test, Y_train, Y_test):
    from sklearn.ensemble import RandomForestRegressor

    # Initializing the Model
    rfr = RandomForestRegressor(n_estimators=50)

    # Fitting the Training Data
    rfr.fit(X_train, Y_train.ravel())

    # Predicting on Test Set
    Y_pred = rfr.predict(X_test)

    # Evaluating the Model
    rfr_ac = r2_score(Y_test, Y_pred)
    print(f'\n\nRANDOM FOREST REGRESSOR\nAccuracy: {rfr_ac}')

    return rfr_ac

if __name__ == '__main__':

    # Importing the Dataset
    dataset = pd.read_csv('housing_data.csv')

    # Preparing the Dataset
    X_train, X_test, Y_train, Y_test = prepare_dataset(dataset)

    # Predicting Using Various Models
    print("\nPredicting Using Various Models:- ")
    lr_ac = linear_regression(X_train, X_test, Y_train, Y_test)
    svr_ac = svr(X_train, X_test, Y_train, Y_test)
    dtr_ac = decision_tree(X_train, X_test, Y_train, Y_test)
    rfr_ac = random_forest(X_train, X_test, Y_train, Y_test)

    # Comparing Accuracy
    print("\nComparing Accuracy of all the Models:- \n")
    print(pd.DataFrame(zip(['Linear Regression', 'SVM Regressor', 'Decision Tree Classifier', 'Random Forest Regressor'],
                           [f'{int(lr_ac * 100)} %', f'{int(svr_ac * 100)} %', f'{int(dtr_ac * 100)} %',
                            f'{int(rfr_ac * 100)} %']),columns=['MODEL','ACCURACY']))
