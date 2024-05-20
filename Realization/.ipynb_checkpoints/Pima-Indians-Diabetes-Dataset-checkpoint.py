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


# Набор данных взят из следующего источника:
# https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database?resource=download

# In[2]:


pima = pd.read_csv("Datasets/pima-indians-diabetes.data.csv", header=None)


# In[3]:


pima


# Набор данных The Pima Indians Diabetes Dataset посвящен прогнозированию развития диабета у индейцев Пима в течение 5 лет с учетом медицинских данных.
# 
# Это задача бинарной классификации. Число наблюдения для классов не сбалансировано. Имеется 768 наблюдений, 8 признаков, одна целевая переменная.
# 
# * Число беременностей.
# 
# * Концентрация глюкозы в крови.
#     
# * Артериальное давление.
# 
# * Толщина кожной складки трицепса.
# 
# * Инсулин.
# 
# * Индекс массы тела.
# 
# * Функция родословной.
# 
# * Возраст (в годах).
# 
# 

# In[4]:


pima.describe()


# In[5]:


pima[[1,2,3,4,5]] =  pima[[1,2,3,4,5]].replace(0, np.NaN)


# In[6]:


pima


# In[7]:


labels = pima[8]


# In[8]:


pima.columns = ['Number of times pregnant',
'Plasma glucose concentration',
'Diastolic blood pressure',
'Triceps skinfold thickness',
'2-Hour serum insulin',
'Body mass index',
'Diabetes pedigree function',
'Age',
'Class variable (0 or 1)']


# In[9]:


F1=pima.drop(columns=['Class variable (0 or 1)'])


# # Корреляция

# In[10]:


F1.corr().round(2)


# # Проверка наличия дизбаланса классов

# In[11]:


sum(labels==0)


# In[12]:


sum(labels==1)


# In[13]:


pima1 = F1.values


# In[14]:


pima1


# In[15]:


Full_Data = {}
Full_Data['mean_fill'] = mvt.mean_fill(pima1)
Full_Data['stoch_lr_fill'] = mvt.lr_fill(pima1, noise=True)
Full_Data['em_fill'] = mvt.EM(pima1)
Full_Data['bootstrap_fill'] = mvt.bootstrap_fill(pima1)
#Full_Data['KNN_fill'] = mvt.knn_fill(pima1)
Full_Data['KNN_fill2'] = mvt.knn_fill(pima1, weights='distance')
Full_Data['iterative_fill_LR'] = mvt.iterative_fill(pima1, 'LR')
Full_Data['iterative_fill_RF'] = mvt.iterative_fill(pima1, 'RF')


# In[16]:


res = solver.estimates(Full_Data, labels).reshape(-1, 1)


# In[17]:


df_index = ['Заполнение средним', 'Стохастическое заполнение линейной регрессией', 'EM-алгоритм', 'Бутстрап', 'Метод ближайших соседей', 'Итеративное заполнение, линейная регрессия', 'Итеративное заполнение, случайный лес', 'LightGBM','Catboost']


# In[18]:


overall_results = pd.DataFrame(res, columns=['F1-мера'], index=df_index)


# In[19]:


overall_results.sort_values('F1-мера',axis=0, ascending=False).round(decimals=3)


# In[ ]:




