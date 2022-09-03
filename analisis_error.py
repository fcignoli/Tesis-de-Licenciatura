#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  5 18:30:23 2022

@author: felipe
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
sns.set_theme(style="darkgrid")

bien_clasificados=[]
suma=np.sum(clases_train_todos_3d)
bien_clasificados= np.concatenate((bien_clasificados,[suma]), axis=0)
suma=np.sum(clases_train_todos_4d)
bien_clasificados= np.concatenate((bien_clasificados,[suma]), axis=0)
suma=np.sum(clases_train_todos_5d)
bien_clasificados= np.concatenate((bien_clasificados,[suma]), axis=0)
suma=np.sum(clases_train_todos_6d)
bien_clasificados= np.concatenate((bien_clasificados,[suma]), axis=0)
suma=np.sum(clases_train_todos_7d)
bien_clasificados= np.concatenate((bien_clasificados,[suma]), axis=0)
suma=np.sum(clases_train_todos_8d)
bien_clasificados= np.concatenate((bien_clasificados,[suma]), axis=0)
suma=np.sum(clases_train_todos_9d)
bien_clasificados= np.concatenate((bien_clasificados,[suma]), axis=0)
suma=np.sum(clases_train_todos_10d)
bien_clasificados= np.concatenate((bien_clasificados,[suma]), axis=0)
suma=np.sum(clases_train_todos_12d)
bien_clasificados= np.concatenate((bien_clasificados,[suma]), axis=0)
suma=np.sum(clases_train_todos_15d)
bien_clasificados= np.concatenate((bien_clasificados,[suma]), axis=0)


dimensiones=[3,4,5,6,7,8,9,10,12,15]
plt.plot(dimensiones, bien_clasificados,'.-')
plt.title('Cantidad de puntos bien clasificados')
plt.xlabel('Dimensiones')
plt.ylabel('# Predicciones')



mal_clasificados=[]
suma=len(clases_train_todos_3d)-np.sum(clases_train_todos_3d)
mal_clasificados= np.concatenate((mal_clasificados,[suma]), axis=0)
suma=len(clases_train_todos_4d)-np.sum(clases_train_todos_4d)
mal_clasificados= np.concatenate((mal_clasificados,[suma]), axis=0)
suma=len(clases_train_todos_5d)-np.sum(clases_train_todos_5d)
mal_clasificados= np.concatenate((mal_clasificados,[suma]), axis=0)
suma=len(clases_train_todos_6d)-np.sum(clases_train_todos_6d)
mal_clasificados= np.concatenate((mal_clasificados,[suma]), axis=0)
suma=len(clases_train_todos_7d)-np.sum(clases_train_todos_7d)
mal_clasificados= np.concatenate((mal_clasificados,[suma]), axis=0)
suma=len(clases_train_todos_8d)-np.sum(clases_train_todos_8d)
mal_clasificados= np.concatenate((mal_clasificados,[suma]), axis=0)
suma=len(clases_train_todos_9d)-np.sum(clases_train_todos_9d)
mal_clasificados= np.concatenate((mal_clasificados,[suma]), axis=0)
suma=len(clases_train_todos_10d)-np.sum(clases_train_todos_10d)
mal_clasificados= np.concatenate((mal_clasificados,[suma]), axis=0)
suma=len(clases_train_todos_12d)-np.sum(clases_train_todos_12d)
mal_clasificados= np.concatenate((mal_clasificados,[suma]), axis=0)
suma=len(clases_train_todos_15d)-np.sum(clases_train_todos_15d)
mal_clasificados= np.concatenate((mal_clasificados,[suma]), axis=0)

dimensiones=[3,4,5,6,7,8,9,10,12,15]

plt.figure()
plt.plot(dimensiones, mal_clasificados,'.-')
plt.title('Cantidad de puntos mal predichos')
plt.xlabel('Dimensiones')
plt.ylabel('# Predicciones')