import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sps
import sklearn.neighbors._base 
from sklearn import linear_model, ensemble
from copy import deepcopy
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer

def prob_intervals(number: int = 100, distribution: str = 'norm')->list:
    '''
    Функция реализует вычисление вероятностей попадания СВ в интервалы, на которые делится основная (99%) часть плотности нормального распределения.
    '''
    if distribution=='norm':
        points=np.linspace(-3, 3, number+1)
        F=sps.norm.cdf(points, loc=0, scale=1)
    if distribution=='beta':
        points=np.linspace(0, 1, number+1)
        F=sps.beta.cdf(points, a=0.9, b=0.9)
    weights=list()
    for i in range(len(F)-1):
        weights.append(abs(F[i+1]-F[i]))
    weights = np.array(weights)
    weights /= sum(weights)
    assert sum(weights)>0.999 and sum(weights)<=1+1e-5
    return weights

def MNAR_norm(X: np.ndarray, mis_cols: object, size_mv: object, rs: int = 42) -> np.ndarray:
    '''
    Функция генерирует неслучайные пропуски; индексы, которым соответствуют пропуски, выбираются с помощью взвешенного случайного выбора, 
    для индекса вероятность быть выбранным определяется нормальным распределением.
    '''
    if type(mis_cols)==int:
        mis_cols=[mis_cols]
    if type(size_mv)==int:
        size_mv=[size_mv]
    assert len(mis_cols)==len(size_mv)
    X1 = X.copy() 
    for i in range(len(mis_cols)):
        p=prob_intervals(number=X.shape[0],distribution='norm')
        x=np.arange(0,len(X))
        chosen_inds=np.random.RandomState(rs).choice(x, size=size_mv[i], replace=False, p=p)
        for j in range(len(X)):
            if j in chosen_inds:
                X1[j,i]=np.nan
    return X1

def MNAR_beta(X: np.ndarray, mis_cols: object, size_mv: object, rs: int = 42) -> np.ndarray:
    '''
    Функция генерирует неслучайные пропуски; индексы, которым соответствуют пропуски, выбираются с помощью взвешенного случайного выбора, 
    для индекса вероятность быть выбранным определяется бета распределением.
    '''
    if type(mis_cols)==int:
        mis_cols=[mis_cols]
    if type(size_mv)==int:
        size_mv=[size_mv]
    assert len(mis_cols)==len(size_mv)
    X1 = X.copy() 
    for i in range(len(mis_cols)):
        p=prob_intervals(number=X.shape[0],distribution='beta')
        x=np.arange(0,len(X))
        chosen_inds=np.random.RandomState(rs).choice(x, size=size_mv[i], replace=False, p=p)
        for j in range(len(X)):
            if j in chosen_inds:
                X1[j,i]=np.nan
    return X1

def MCAR(X: np.ndarray, mis_cols: object, size_mv: object , rs: int = 42) -> np.ndarray:
    '''
    Функция реализует создание неслучайных пропусков, пропуски произвольной переменной не зависят от наблюдаемых или пропущенных значений.
    '''
    if type(mis_cols)==int:
        mis_cols=[mis_cols]
    if type(size_mv)==int:
        size_mv=[size_mv]
    assert len(mis_cols)==len(size_mv)
    X1 = X.copy()
    for i in range(len(mis_cols)):
        h = np.array([1]*size_mv[i]+[0]*(len(X)-size_mv[i]))
        np.random.RandomState(rs).shuffle(h)
        X1[:,mis_cols[i]][np.where(h==1)] = np.nan
    return X1

def MAR(X: np.ndarray, indicator: int, mis_cols: object, thresh: int, to_remove: str) -> np.ndarray:
    '''
    Функция реализует создание случайных пропусков, пропуски произвольной переменной зависят от наблюдаемых данных.
    '''
    if type(mis_cols)==int:
        mis_cols=[mis_cols]
    assert indicator not in mis_cols
    X1 = X.copy()
    for row in range(len(X)):
        if (to_remove=='>' and X1[row,indicator]>=thresh) or (to_remove=='<' and X1[row,indicator]<=thresh):
            X1[row,mis_cols] = np.nan  
    return X1

def MNAR(X: np.ndarray, mis_cols: dict, to_remove: str) -> np.ndarray:
    '''
    Функция реализует создание неслучайных пропусков, пропуски произвольной переменной зависят от пропущенного значения.
    '''
    X1 = X.copy()
    for row in range(len(X)):
        for col in mis_cols.keys():
            if (to_remove=='>' and X1[row,col]>=mis_cols[col]) or (to_remove=='<' and X1[row,col]<=mis_cols[col]):
                X1[row,col] = np.nan               
    return X1

def groups_of_indices(X: np.ndarray) -> tuple([list, list]):
    '''
    Функция осуществляет разбиение индексов столбцов датасета на две группы на основе наличия пропусков в столбцах.
    '''
    obs_cols, mis_cols = [], []
    for col in range(len(X[0])):
        if np.isnan(np.sum(X[:,col])):
            mis_cols.append(col)
        else:
            obs_cols.append(col)
    return obs_cols, mis_cols

def mean_fill(X: np.ndarray) -> np.ndarray:
    '''
    Функция реализует заполнение пропусков средним значением.
    '''
    X1 = X.copy()
    obs_cols, mis_cols = groups_of_indices(X)
    for col in mis_cols:
        m = np.nanmean(X1[:,col])
        X1[:,col][np.isnan(X1[:,col])] = m
    return X1

def lr_fill(X: np.ndarray, noise: bool = False, rs: int = 42) -> np.ndarray:
    '''
    Функция реализует заполнение пропусков с использованием линейной регрессии.
    '''
    X1 = X.copy()
    obs_cols, mis_cols = groups_of_indices(X1)
    for col in mis_cols:
        reg = linear_model.LinearRegression()
        not_nan_rows, nan_rows = np.where(~np.isnan(X1[:,col])), np.where(np.isnan(X1[:,col]))
        train_data, train_labels, test_data = X1[not_nan_rows][:,obs_cols], X1[not_nan_rows][:,col], X1[nan_rows][:,obs_cols]
        if train_data.shape==(len(train_data),):
            train_data = train_data.reshape(-1,1)
        if test_data.shape==(len(test_data),):
            test_data = test_data.reshape(-1,1)
        reg.fit(train_data, train_labels)
        sigma = np.sqrt(np.var(train_labels-reg.predict(train_data)))
        X1[:,col][np.isnan(X1[:,col])] = reg.predict(test_data)+noise*np.random.RandomState(rs).normal(0, sigma, size=len(nan_rows))
    return X1

def bootstrap_fill(X: np.ndarray, rs: int = 42) -> np.ndarray:
    '''
    Функция реализует заполнение пропусков с использованием бутстрапа (создания выборок с возвращением из наблюдаемых данных)
    '''
    X1 = X.copy()
    obs_cols, mis_cols = groups_of_indices(X1)
    for col in mis_cols:
        sample = X1[:,col][~np.isnan(X1[:,col])]
        mv_count = len(X1[:,col])-len(sample)
        X1[:,col][np.isnan(X1[:,col])] = np.random.RandomState(rs).choice(sample, mv_count)
    return X1

def EM(X: np.ndarray, max_iter: int = 20, rtol: float = 1e-8) -> np.ndarray:
    '''
    Функция применяет алгоритм максимального правдоподобия к полученным данным для восстановления пропущенных значений.
    '''
    Indicator = np.isnan(X)
    col_numbers = np.arange(1, X.shape[1] + 1)
    M, O = col_numbers * Indicator - 1, col_numbers * (Indicator == False) - 1
    Mu = np.nanmean(X, axis = 0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    K = np.cov(X[observed_rows, ].T)
    if np.isnan(K).any():
        K = np.diag(np.nanvar(X, axis = 0))
    Mu_cond, K_cond_accum = {}, np.zeros((X.shape[1], X.shape[1]))
    X_modified = X.copy()
    EM_Iteration = 0
    while EM_Iteration < max_iter:
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(col_numbers - 1):
                M_i, O_i = M[i, ][M[i, ] > -1], O[i, ][O[i, ] > -1]
                K_MM, K_MO, K_OO = K[np.ix_(M_i, M_i)], K[np.ix_(M_i, O_i)], K[np.ix_(O_i, O_i)]
                K_OM = K_MO.T
                Mu_cond[i] = Mu[np.ix_(M_i)] + K_MO @ np.linalg.inv(K_OO) @ (X_modified[i, O_i] - Mu[np.ix_(O_i)])
                X_modified[i, M_i] = Mu_cond[i]
                K_cond = K_MM - K_MO @ np.linalg.inv(K_OO) @ K_OM
                K_cond_accum[np.ix_(M_i, M_i)] += K_cond
        Mu_new, K_new = np.mean(X_modified, axis = 0), np.cov(X_modified.T, bias = 1) + K_cond_accum / X.shape[0]
        if np.linalg.norm(Mu - Mu_new) < rtol and np.linalg.norm(K - K_new, ord = 2) < rtol:
            break
        Mu, K = Mu_new, K_new
        for i in range(K.shape[0]):
            assert K[i,i]>=0, f'Variance of {i} feature on iteration {EM_Iteration} is negative'
            assert np.linalg.det(K)>=0, f'Determinant of Covariance matrix on iteration {EM_Iteration} is negative'
        EM_Iteration += 1
    return X_modified

def iterative_fill(X: np.ndarray, suggested_model='RF') -> np.ndarray:
    '''
    Функция применяет алгоритм IterativeImputer к полученным данным для заполнения пропусков, с использованием предложенной модели.
    '''
    if suggested_model=='RF':
        model = ensemble.RandomForestRegressor(n_estimators=200, random_state=0)
    if suggested_model=='LR':
        model = linear_model.LinearRegression()
    myImputer=IterativeImputer(estimator=model, random_state=0, max_iter=5)
    myImputer.fit(X)
    return myImputer.transform(X)
           
def knn_fill(X: np.ndarray, n: int = 5, weights: str = 'uniform') -> np.ndarray:
    '''
    Функция применяет метод ближайших соседей к полученным данным для заполнения пропусков. 
    '''
    imputer = KNNImputer(n_neighbors=n, weights=weights)
    return imputer.fit_transform(X)