import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as sps
import sklearn.neighbors._base 
import sys
sys.modules['sklearn.neighbors.base'] = sklearn.neighbors._base
# Модуль ниже требует sklearn=1.1.2, scipy=1.9.1
from missingpy import MissForest
from functools import reduce
from itertools import chain, combinations
from sklearn import linear_model
from copy import deepcopy
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from functools import lru_cache

# Вычисление вероятностей попадания СВ в интервалы, на которые делится плотность вероятности.
def prob_intervals(number=100,distribution='norm')->list:
    if distribution=='norm':
        points=np.linspace(-4, 4, number+1)
        F=sps.norm.cdf(points, loc=0, scale=1)
    if distribution=='beta':
        points=np.linspace(0, 1, number+1)
        F=sps.beta.cdf(points, a=0.9, b=0.9)
    weights=list()
    for i in range(len(F)-1):
        weights.append(abs(F[i+1]-F[i]))
    assert sum(weights)>0.999 and sum(weights)<=1+1e-5
    if sum(weights)<1:
        weights[0]=weights[0]+1-sum(weights)
    return weights

#Генерация неслучайных данных, завимость от индекса в соответствии с нормальным распределением
def MNAR_norm(X: np.ndarray, mis_cols, size_mv, rs=42) -> np.ndarray:
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

#Генерация неслучайных данных, завимость от индекса в соответствии с бета-распределением
def MNAR_beta(X: np.ndarray, mis_cols, size_mv, rs=42) -> np.ndarray:
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

# Генерация совершенно случайных данных
def MCAR(X: np.ndarray, mis_cols, size_mv , rs=42) -> np.ndarray:
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

# Генерация случайных данных
def MAR(X: np.ndarray, indicator: int, mis_cols, thresh: int, to_remove: str) -> np.ndarray:
    if type(mis_cols)==int:
        mis_cols=[mis_cols]
    assert indicator not in mis_cols
    X1 = X.copy()
    for row in range(len(X)):
        if (to_remove=='>' and X1[row,indicator]>=thresh) or (to_remove=='<' and X1[row,indicator]<=thresh):
            X1[row,mis_cols] = np.nan  
    return X1

#Генерация неслучайных данных
def MNAR(X: np.ndarray, mis_cols: dict, to_remove: str) -> np.ndarray:
    X1 = X.copy()
    for row in range(len(X)):
        for col in mis_cols.keys():
            if (to_remove=='>' and X1[row,col]>=mis_cols[col]) or (to_remove=='<' and X1[row,col]<=mis_cols[col]):
                X1[row,col] = np.nan               
    return X1

# Разбиение индексов на две группы: одни соответствуют столбцам без пропусков, другие соответствуют столбцам с пропусками
def groups_of_indices(X: np.ndarray) -> tuple([list, list]):
    obs_cols = []
    mis_cols = []
    for col in range(len(X[0])):
        if np.isnan(np.sum(X[:,col])):
            mis_cols.append(col)
        else:
            obs_cols.append(col)
    return obs_cols, mis_cols

# Заполнение средним
def mean_fill(X: np.ndarray) -> np.ndarray:
    X1 = X.copy()
    obs_cols, mis_cols = groups_of_indices(X)
    for col in mis_cols:
        m = np.nanmean(X1[:,col])
        X1[:,col][np.isnan(X1[:,col])] = m
    return X1

# Заполнение по регрессии
def lr_fill(X: np.ndarray, noise=False, rs=42) -> np.ndarray:
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

# Заполнение с помощью бутстрапа
def bootstrap_fill(X: np.ndarray, rs=42) -> np.ndarray:
    X1 = X.copy()
    obs_cols, mis_cols = groups_of_indices(X1)
    for col in mis_cols:
        sample = X1[:,col][~np.isnan(X1[:,col])]
        mv_count = len(X1[:,col])-len(sample)
        X1[:,col][np.isnan(X1[:,col])] = np.random.RandomState(rs).choice(sample, mv_count)
    return X1

# EM-алгоритм
def EM(X:np.ndarray, max_iter=20, rtol=1e-8) -> np.ndarray:
    C = np.isnan(X)==False
    one_to_ncol = np.arange(1, X.shape[1] + 1)
    M = one_to_ncol * (C == False) - 1
    O = one_to_ncol * C - 1
    Mu = np.nanmean(X, axis = 0)
    observed_rows = np.where(np.isnan(sum(X.T)) == False)[0]
    S = np.cov(X[observed_rows, ].T)
    if np.isnan(S).any():
        S = np.diag(np.nanvar(X, axis = 0))
    Mu_tilde, S_tilde = {}, np.zeros((X.shape[1], X.shape[1]))
    X_modified = X.copy()
    iteration = 0
    while iteration < max_iter:
        for i in range(X.shape[0]):
            if set(O[i, ]) != set(one_to_ncol - 1):
                M_i, O_i = M[i, ][M[i, ] != -1], O[i, ][O[i, ] != -1]
                S_MM = S[np.ix_(M_i, M_i)]
                S_MO = S[np.ix_(M_i, O_i)]
                S_OM = S_MO.T
                S_OO = S[np.ix_(O_i, O_i)]
                Mu_tilde[i] = Mu[np.ix_(M_i)] +\
                    S_MO @ np.linalg.inv(S_OO) @\
                    (X_modified[i, O_i] - Mu[np.ix_(O_i)])
                X_modified[i, M_i] = Mu_tilde[i]
                S_MM_O = S_MM - S_MO @ np.linalg.inv(S_OO) @ S_OM
                S_tilde[np.ix_(M_i, M_i)] += S_MM_O
        Mu_new = np.mean(X_modified, axis = 0)
        S_new = np.cov(X_modified.T, bias = 1) + S_tilde / X.shape[0]
        if np.linalg.norm(Mu - Mu_new) < rtol and np.linalg.norm(S - S_new, ord = 2) < rtol:
            break
        Mu = Mu_new
        S = S_new
        for i in range(S.shape[0]):
            assert S[i,i]>=0, f'Variance of {i} feature on iteration {iteration} is negative'
            assert np.linalg.det(S)>=0, f'Determinant of Covariance matrix on iteration {iteration} is negative'
        iteration += 1
    return X_modified

# Множественное вменение
def multiple_fill(X: np.ndarray) -> np.ndarray:
    myImputer=IterativeImputer(random_state=42)
    myImputer.fit(X)
    return myImputer.transform(X)
           
# KNN-метод
def knn_fill(X: np.ndarray, n=5, weights='uniform') -> np.ndarray:
    imputer = KNNImputer(n_neighbors=n, weights=weights)
    return imputer.fit_transform(X)

# Miss-forest
def missforest_fill(X: np.ndarray, max_iter=15, n_estimators=100) -> np.ndarray:
    mf = MissForest(max_iter, n_estimators)
    return mf.fit_transform(X)
    
# Создаем множество всех подмножеств из iterable, чья длина не меньше min_subset_len
def powerset(iterable, min_subset_len=2) -> list:
    s = list(iterable)
    return list(chain.from_iterable(combinations(s, r) for r in range(min_subset_len,len(s)+1)))

# Отбор признаков, выбираем подмножество с самым большим по модулю определителем, указываем минимальный размер подможества признаков
def feature_choose(df:pd.DataFrame, min_subset:int) -> tuple:
    cols=df.columns
    corr_dets = []
    cols_power_set=powerset(cols,min_subset)
    for item in cols_power_set:
        corr_dets.append((item,np.linalg.det(df[list(item)].corr())))
    min_corr_det_ind, min_corr_det = -1, 0
    for i, item in enumerate(corr_dets):
        if abs(item[1])>abs(min_corr_det):
            min_corr_det, min_corr_det_ind = item[1], i     
    return corr_dets[min_corr_det_ind] 
    