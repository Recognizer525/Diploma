#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Импортируем необходимые библиотеки
from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
#Новые библиотеки с выделенными функциями
import mv_tools as mvt
import solver
warnings.filterwarnings('ignore')


# Датасет взят из следующего источника: https://www.openml.org/search?type=data&sort=runs&status=active&order=desc&qualities.NumberOfClasses=%3D_2&qualities.NumberOfFeatures=lte_10&qualities.NumberOfInstances=lte_1000&id=1005,
# также может быть найден (в ином формате) по следующей ссылке: https://archive.ics.uci.edu/dataset/42/glass+identification

# # Считываем датасет из файла, отдельно метки, отдельно признаки

# In[2]:


features = []
with open('Datasets/glass.arff','r') as f1:
    for line in f1:
        if line[0].isdigit():
            if line[-1]=='\n':
                data = line[:-1].split(',')
            else:
                data = line.split(',')
            features.append(data)
            for i in range(len(features[-1])):
                try:
                    features[-1][i]=float(features[-1][i])
                except:
                    pass
                    
F=pd.DataFrame(features, columns=['RI','NA','Mg','Al','Si','K','Ca','Ba','Fe','labels'])  


# In[3]:


F


# In[4]:


F1=F.drop(columns=['labels'])


# In[5]:


F1.corr().round(2)


# In[6]:


np.linalg.det(F1.corr().round(2).values)


# In[7]:


dataplot = sns.heatmap(F1.corr().round(2), cmap="crest", annot=True)


# # Тест на нормальное распределение признаков

# In[8]:


for col in F1.columns:
    print(stats.shapiro(F1[col]))


# In[9]:


F_np=F1.values


# In[10]:


log_F = np.log(F_np+np.full(F_np.shape, 1e-6))


# In[11]:


log_F


# In[12]:


for i in range(len(log_F[0])):
    print(stats.shapiro(log_F[:,i]))


# In[13]:


F_labels = list(F['labels'])
for i in range(len(F_labels)):
    if F_labels[i]=="N":
        F_labels[i] = 0
    if F_labels[i]=="P":
        F_labels[i] = 1


# In[14]:


print(f"Число разных классов для датасета: {len(set(F_labels))}")


# # Оценим, какая модель наилучшим образом решит задачу классификации для данного набора данных.

# In[15]:


initial_results = solver.initial_estimates(F_np, F_labels)
methods = ['Случайный лес','CatBoost','LightGBM','Логистическая регрессия','AdaBoost']
initial_results_dict = dict(zip(methods,initial_results))


# In[16]:


initial_results_dict


# In[17]:


initial_results_df=pd.DataFrame(np.array(initial_results).reshape(-1,1), columns=['F1-мера'], index=methods)


# In[18]:


initial_results_df.sort_values(by=['F1-мера'],ascending=False).round(decimals=3)


# # Добавим пропуски в данные (10% по столбцам 2-5, 8-9)

# In[19]:


# Создаем версию датасета с абсолютно случайными пропусками, т.е. независящими от наблюдаемых данных, 10% пропусков
D_MCAR = mvt.MCAR(F_np, [1,2,3,4,7,8], [int(1*len(F_np)/10)]*6)
Full_D_MCAR = {}
Full_D_MCAR['mean_fill'] = mvt.mean_fill(D_MCAR)
Full_D_MCAR['stoch_lr_fill'] = mvt.lr_fill(D_MCAR, noise=True)
Full_D_MCAR['em_fill'] = mvt.EM(D_MCAR)
Full_D_MCAR['bootstrap_fill'] = mvt.bootstrap_fill(D_MCAR)
Full_D_MCAR['KNN_fill'] = mvt.knn_fill(D_MCAR)
#Full_D_MCAR['KNN_fill2'] = mvt.knn_fill(D_MCAR, weights='distance')
Full_D_MCAR['iterative_fill_LR'] = mvt.iterative_fill(D_MCAR, 'LR')
Full_D_MCAR['iterative_fill_RF'] = mvt.iterative_fill(D_MCAR, 'RF')


# In[20]:


# Создаем версию датасета со случайными пропусками, x_i2 становится пропуском, если x_i1 больше квантиля 0.9 для признака x1
D_MAR = mvt.MAR(F_np, indicator=0, mis_cols=[1,2,3,4,7,8], thresh=np.quantile(F_np[:,0], 0.9), to_remove='>')
Full_D_MAR = {}
Full_D_MAR['mean_fill'] = mvt.mean_fill(D_MAR)
Full_D_MAR['stoch_lr_fill'] = mvt.lr_fill(D_MAR, noise=True)
Full_D_MAR['em_fill'] = mvt.EM(D_MAR)
Full_D_MAR['bootstrap_fill'] = mvt.bootstrap_fill(D_MAR)
Full_D_MAR['KNN_fill'] = mvt.knn_fill(D_MAR)
#Full_D_MAR['KNN_fill2'] = mvt.knn_fill(D_MAR, weights='distance')
Full_D_MAR['iterative_fill_LR'] = mvt.iterative_fill(D_MAR, 'LR')
Full_D_MAR['iterative_fill_RF'] = mvt.iterative_fill(D_MAR, 'RF')


# In[21]:


# Создаем версию датасета с неслучайными пропусками, 
#пропуски распределены в соответствии с нормальным распределением относительно номера строки, 10% пропусков
D_MNAR_norm = mvt.MNAR_norm(F_np, [1,2,3,4,7,8], [int(1*len(F_np)/10)]*6)
Full_D_MNAR_norm = {}
Full_D_MNAR_norm['mean_fill'] = mvt.mean_fill(D_MNAR_norm)
Full_D_MNAR_norm['stoch_lr_fill'] = mvt.lr_fill(D_MNAR_norm, noise=True)
Full_D_MNAR_norm['em_fill'] = mvt.EM(D_MNAR_norm)
Full_D_MNAR_norm['bootstrap_fill'] = mvt.bootstrap_fill(D_MNAR_norm)
Full_D_MNAR_norm['KNN_fill'] = mvt.knn_fill(D_MNAR_norm)
#Full_D_MNAR_norm['KNN_fill2'] = mvt.knn_fill(D_MNAR_norm, weights='distance')
Full_D_MNAR_norm['iterative_fill_LR'] = mvt.iterative_fill(D_MNAR_norm, 'LR')
Full_D_MNAR_norm['iterative_fill_RF'] = mvt.iterative_fill(D_MNAR_norm, 'RF')


# In[22]:


# Создаем версию датасета с неслучайными пропусками, 
#пропуски распределены в соответствии с нормальным распределением относительно номера строки, 10% пропусков
D_MNAR_beta = mvt.MNAR_beta(F_np, [1,2,3,4,7,8], [int(1*len(F_np)/10)]*6)
Full_D_MNAR_beta = {}
Full_D_MNAR_beta['mean_fill'] = mvt.mean_fill(D_MNAR_beta)
Full_D_MNAR_beta['stoch_lr_fill'] = mvt.lr_fill(D_MNAR_beta, noise=True)
Full_D_MNAR_beta['em_fill'] = mvt.EM(D_MNAR_beta)
Full_D_MNAR_beta['bootstrap_fill'] = mvt.bootstrap_fill(D_MNAR_beta)
Full_D_MNAR_beta['KNN_fill'] = mvt.knn_fill(D_MNAR_beta)
#Full_D_MNAR_beta['KNN_fill2'] = mvt.knn_fill(D_MNAR_beta, weights='distance')
Full_D_MNAR_beta['iterative_fill_LR'] = mvt.iterative_fill(D_MNAR_beta, 'LR')
Full_D_MNAR_beta['iterative_fill_RF'] = mvt.iterative_fill(D_MNAR_beta, 'RF')


# In[23]:


df_index = ['Заполнение средним', 'Стохастическое заполнение линейной регрессией', 'EM-алгоритм', 'Бутстрап', 'Метод ближайших соседей', 'Итеративное заполнение, линейная регрессия', 'Итеративное заполнение, случайный лес', 'LightGBM','Catboost']
df_index2 = ['Заполнение средним', 'Стохастическое заполнение линейной регрессией', 'EM-алгоритм', 'Бутстрап', 'Метод ближайших соседей', 'Метод ближайших соседей, взвешенный', 'Итеративное заполнение, линейная регрессия', 'Итеративное заполнение, случайный лес', 'LightGBM','Catboost']
df_columns = ['MCAR','MAR','MNAR N(0,1)','MNAR B(0.9,0.9)']


# # Решим задачу классификации, чтобы оценить качество заполнения.

# In[24]:


MCAR_column = solver.estimates(Full_D_MCAR, F_labels).reshape(-1,1)
MAR_column = solver.estimates(Full_D_MAR, F_labels).reshape(-1,1)
MNAR_norm_column = solver.estimates(Full_D_MNAR_norm, F_labels).reshape(-1,1)
MNAR_beta_column = solver.estimates(Full_D_MNAR_beta, F_labels).reshape(-1,1)


# In[25]:


values=np.hstack((MCAR_column, MAR_column, MNAR_norm_column, MNAR_beta_column))


# In[26]:


overall_results=pd.DataFrame(values, columns=df_columns, index=df_index)


# In[27]:


overall_results.round(decimals=3)


# In[28]:


proximity_array = np.zeros((len(df_index)-2, len(df_columns)))
for i, item in enumerate(Full_D_MCAR.keys()):
    proximity_array[i,0] = solver.proximity_estimate(F_np, Full_D_MCAR[item])
for i, item in enumerate(Full_D_MAR.keys()):
    proximity_array[i,1] = solver.proximity_estimate(F_np, Full_D_MAR[item])
for i, item in enumerate(Full_D_MNAR_norm.keys()):
    proximity_array[i,2] = solver.proximity_estimate(F_np, Full_D_MNAR_norm[item])
for i, item in enumerate(Full_D_MNAR_beta.keys()):
    proximity_array[i,3] = solver.proximity_estimate(F_np, Full_D_MNAR_beta[item])
df_proximity = pd.DataFrame(proximity_array, columns=df_columns, index=df_index[:-2])


# In[29]:


df_proximity.round(decimals=3)


# # Увеличим число пропусков до 60%

# In[30]:


# Создаем версию датасета с абсолютно случайными пропусками, т.е. независящими от наблюдаемых данных, 60% пропусков
D_MCAR2 = mvt.MCAR(F_np,  [1,2,3,4,7,8], [int(6*len(F_np)/10)]*6)
Full_D_MCAR2 = {}
Full_D_MCAR2['mean_fill'] = mvt.mean_fill(D_MCAR2)
Full_D_MCAR2['stoch_lr_fill'] = mvt.lr_fill(D_MCAR2, noise=True)
Full_D_MCAR2['em_fill'] = mvt.EM(D_MCAR2)
Full_D_MCAR2['bootstrap_fill'] = mvt.bootstrap_fill(D_MCAR2)
Full_D_MCAR2['KNN_fill'] = mvt.knn_fill(D_MCAR2)
#Full_D_MCAR2['KNN_fill2'] = mvt.knn_fill(D_MCAR2, weights='distance')
Full_D_MCAR2['iterative_fill_LR'] = mvt.iterative_fill(D_MCAR2, 'LR')
Full_D_MCAR2['iterative_fill_RF'] = mvt.iterative_fill(D_MCAR2, 'RF')


# In[31]:


# Создаем версию датасета со случайными пропусками, x_i2 становится пропуском, если x_i1 больше квантиля 0.6 для признака x1
D_MAR2 = mvt.MAR(F_np, indicator=0, mis_cols=[1,2,3,4,7,8], thresh=np.quantile(F_np[:,0], 0.6), to_remove='>')
Full_D_MAR2 = {}
Full_D_MAR2['mean_fill'] = mvt.mean_fill(D_MAR2)
Full_D_MAR2['stoch_lr_fill'] = mvt.lr_fill(D_MAR2, noise=True)
Full_D_MAR2['em_fill'] = mvt.EM(D_MAR2)
Full_D_MAR2['bootstrap_fill'] = mvt.bootstrap_fill(D_MAR2)
Full_D_MAR2['KNN_fill'] = mvt.knn_fill(D_MAR2)
#Full_D_MAR2['KNN_fill2'] = mvt.knn_fill(D_MAR2, weights='distance')
Full_D_MAR2['iterative_fill_LR'] = mvt.iterative_fill(D_MAR2, 'LR')
Full_D_MAR2['iterative_fill_RF'] = mvt.iterative_fill(D_MAR2, 'RF')


# In[32]:


# Создаем версию датасета с неслучайными пропусками, 
#пропуски распределены в соответствии с нормальным распределением относительно номера строки, 60% пропусков
D_MNAR_norm2 = mvt.MNAR_norm(F_np,  [1,2,3,4,7,8], [int(6*len(F_np)/10)]*6)
Full_D_MNAR_norm2 = {}
Full_D_MNAR_norm2['mean_fill'] = mvt.mean_fill(D_MNAR_norm2)
Full_D_MNAR_norm2['stoch_lr_fill'] = mvt.lr_fill(D_MNAR_norm2, noise=True)
Full_D_MNAR_norm2['em_fill'] = mvt.EM(D_MNAR_norm2)
Full_D_MNAR_norm2['bootstrap_fill'] = mvt.bootstrap_fill(D_MNAR_norm2)
Full_D_MNAR_norm2['KNN_fill'] = mvt.knn_fill(D_MNAR_norm2)
#Full_D_MNAR_norm2['KNN_fill2'] = mvt.knn_fill(D_MNAR_norm2, weights='distance')
Full_D_MNAR_norm2['iterative_fill_LR'] = mvt.iterative_fill(D_MNAR_norm2, 'LR')
Full_D_MNAR_norm2['iterative_fill_RF'] = mvt.iterative_fill(D_MNAR_norm2, 'RF')


# In[33]:


# Создаем версию датасета с неслучайными пропусками, 
#пропуски распределены в соответствии с нормальным распределением относительно номера строки, 60% пропусков
D_MNAR_beta2 = mvt.MNAR_beta(F_np,  [1,2,3,4,7,8], [int(6*len(F_np)/10)]*6)
Full_D_MNAR_beta2 = {}
Full_D_MNAR_beta2['mean_fill'] = mvt.mean_fill(D_MNAR_beta2)
Full_D_MNAR_beta2['stoch_lr_fill'] = mvt.lr_fill(D_MNAR_beta2, noise=True)
Full_D_MNAR_beta2['em_fill'] = mvt.EM(D_MNAR_beta2)
Full_D_MNAR_beta2['bootstrap_fill'] = mvt.bootstrap_fill(D_MNAR_beta2)
Full_D_MNAR_beta2['KNN_fill'] = mvt.knn_fill(D_MNAR_beta2)
#Full_D_MNAR_beta2['KNN_fill2'] = mvt.knn_fill(D_MNAR_beta2, weights='distance')
Full_D_MNAR_beta2['iterative_fill_LR'] = mvt.iterative_fill(D_MNAR_beta2, 'LR')
Full_D_MNAR_beta2['iterative_fill_RF'] = mvt.iterative_fill(D_MNAR_beta2, 'RF')


# # Решим задачу классификации, чтобы оценить качество заполнения.

# In[34]:


MCAR_column2 = solver.estimates(Full_D_MCAR2, F_labels).reshape(-1,1)
MAR_column2 = solver.estimates(Full_D_MAR2, F_labels).reshape(-1,1)
MNAR_norm_column2 = solver.estimates(Full_D_MNAR_norm2, F_labels).reshape(-1,1)
MNAR_beta_column2 = solver.estimates(Full_D_MNAR_beta2, F_labels).reshape(-1,1)


# In[35]:


values2=np.hstack((MCAR_column2,MAR_column2,MNAR_norm_column2,MNAR_beta_column2))


# In[36]:


overall_results2=pd.DataFrame(values2, columns=df_columns, index=df_index)


# In[37]:


overall_results2.round(decimals=3)
#dfStyler = overall_results2.style.set_properties(**{'text-align': 'left'})
#dfStyler.set_table_styles([dict(selector='th', props=[('text-align', 'left')])])


# In[38]:


proximity_array2 = np.zeros((len(df_index)-2,len(df_columns)))
for i, item in enumerate(Full_D_MCAR2.keys()):
    proximity_array2[i,0] = solver.proximity_estimate(F_np, Full_D_MCAR2[item])
for i, item in enumerate(Full_D_MAR2.keys()):
    proximity_array2[i,1] = solver.proximity_estimate(F_np, Full_D_MAR2[item])
for i, item in enumerate(Full_D_MNAR_norm2.keys()):
    proximity_array2[i,2] = solver.proximity_estimate(F_np, Full_D_MNAR_norm2[item])
for i, item in enumerate(Full_D_MNAR_beta2.keys()):
    proximity_array2[i,3] = solver.proximity_estimate(F_np, Full_D_MNAR_beta2[item])
df_proximity2 = pd.DataFrame(proximity_array2, columns=df_columns, index=df_index[:-2])


# In[39]:


df_proximity2.round(decimals=3)


# In[ ]:




