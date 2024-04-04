import numpy as np
import warnings
import pandas as pd
import catboost
import lightgbm
import warnings
from sklearn import linear_model, metrics, ensemble
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate, GridSearchCV
warnings.filterwarnings('ignore')

def cls_estimator_cv(Data: np.ndarray, labels: list, model: str = 'RF') -> np.float32:
    '''
    Функция реализует решения задачи классификации для выбранных входных данных, меток и модели.
    '''
    scoring = ['f1']
    if model == 'RF':
        param_grid = { 'n_estimators': [150,200,250,300], 
                      'max_depth': [3, 6, 9, None], 
                     } 
        res = GridSearchCV(ensemble.RandomForestClassifier(random_state=42),
                      param_grid=param_grid,
                      scoring='f1',
                      cv=5)
        res.fit(Data, labels)
        print(f'Параметры для случайного леса: {res.best_params_}')
        clsf = ensemble.RandomForestClassifier(**res.best_params_, random_state=42)
    if model=='LightGBM':
        param_grid = { 'n_estimators': [200,300,500], 
                      'max_depth': [3, 6, 9],
                      'learning_rate':[0.01,0.1],
                     } 
        res = GridSearchCV(lightgbm.LGBMClassifier(random_state=42, verbose=-1),
                      param_grid=param_grid,
                      scoring='f1',
                      cv=5)
        res.fit(Data, labels)
        print(f'Параметры для LightGBM: {res.best_params_}')
        clsf = lightgbm.LGBMClassifier(**res.best_params_, random_state=42, verbose=-1)
    if model=='CatBoost':
        param_grid = { 'iterations': [200,300,500],
                      'depth': [4, 5, 6],
                      'learning_rate': [0.01,0.1],
                     } 
        res = GridSearchCV(catboost.CatBoostClassifier(random_state=42,verbose=False),
                      param_grid=param_grid,
                      scoring='f1',
                      cv=5)
        res.fit(Data, labels)
        print(f'Параметры для CatBoost: {res.best_params_}')
        clsf = catboost.CatBoostClassifier(**res.best_params_, random_state=42,verbose=False)
    if model == 'LogRegression':
        param_grid = { 'C': [100,200,300], 
                      'max_iter': [50,100,150,200],
                     } 
        res = GridSearchCV(linear_model.LogisticRegression(random_state=42),
                      param_grid=param_grid,
                      scoring='f1',
                      cv=5)
        res.fit(Data, labels)
        print(f'Параметры для LogisticRegression: {res.best_params_}')
        clsf = linear_model.LogisticRegression(**res.best_params_,
                                               random_state=42)
    if model == 'AdaBoost':
        param_grid = {'n_estimators': [100,150,200,250,300],
                     } 
        res = GridSearchCV(ensemble.AdaBoostClassifier(random_state=42),
                      param_grid=param_grid,
                      scoring='f1',
                      cv=5)
        res.fit(Data, labels)
        print(f'Параметры для AdaBoost: {res.best_params_}')
        clsf = ensemble.AdaBoostClassifier(**res.best_params_, 
                                           random_state=42)
    scores = cross_val_score(clsf, Data, labels, cv=3, scoring='f1')
    return np.mean(scores)

def final_cls_estimator_cv(Data: np.ndarray, labels: list) -> np.float32:
    '''
    Функция реализует решение задачи классификации для выбранных данных и меток, функция возвращает усредненное значение f1-меры, полученное по итогам
    классификации.
    '''
    clsf = lightgbm.LGBMClassifier(learning_rate=0.1, max_depth=9, n_estimators=300, random_state=42, verbose=-1)
    scores = cross_val_score(clsf, Data, labels, cv=5, scoring='f1')
    return np.mean(scores)

def CV_boosting_clf(Data: np.ndarray, labels: list, model: str = 'LightGBM') -> np.float32:
    '''
    Функция вызывает градиентный бустинг для решения задачи классификации на основе полученных данных. Используется кросс-валидация.
    Функция возвращает усредненное значение f1-меры, полученное по итогам классификации.
    '''
    scoring = ['f1']
    if model == 'LightGBM':
        lgbm_cl = lightgbm.LGBMClassifier(n_estimators=30, random_state=123, verbose=-1)
        scores = cross_validate(lgbm_cl, Data, labels, cv=3, scoring=scoring)
    if model == 'Catboost':
        cat_cl = catboost.CatBoostClassifier(n_estimators=30, random_state=123,verbose=False)
        scores = cross_validate(cat_cl, Data, labels, cv=3, scoring=scoring)
    return np.mean(scores['test_f1'])

def estimates(Data: dict, labels: list) -> np.ndarray:
    '''
    Функция принимает на вход датасет и его метки, функция возвращает метрики качества, полученные в результате решения задачи классификации разными способами.
    '''
    results = np.zeros(len(Data)+2)
    for i, key in enumerate(Data.keys()):
        results[i] = final_cls_estimator_cv(Data[key], labels)
    results[-2] = CV_boosting_clf(Data[key],labels,model='LightGBM')
    results[-1] = CV_boosting_clf(Data[key],labels,model='Catboost')  
    return results

def initial_estimates(Data: np.ndarray, labels: list) -> np.ndarray:
    '''
    Функция принимает на вход датасет и его метки, вызывает алгоритмы классификации, возвращает поулченные значения метрики качества.
    '''
    models = ['RF','CatBoost','LightGBM','LogRegression','AdaBoost']
    results = list()
    for model in models:
        results.append(cls_estimator_cv(Data, labels, model))
    return results

def proximity_estimate(Initial_Data: np.ndarray, Restored_Data: np.ndarray) -> np.float64:
    '''
    Функция вычисляет оценку приближенности результатов заполнения данных.
    '''
    A = np.sum(abs(Initial_Data-Restored_Data)**2)
    matrix_size = Initial_Data.shape[0]*Initial_Data.shape[1]
    return A/matrix_size










