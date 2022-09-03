#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 20:20:36 2022

@author: felipe
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df1=pd.read_csv('/home/felipe/Documentos/tesis/corridas/sin_permutar')
df2=pd.read_csv('/home/felipe/Documentos/tesis/corridas/permutacion1')
df3=pd.read_csv('/home/felipe/Documentos/tesis/corridas/permutacion2')
df4=pd.read_csv('/home/felipe/Documentos/tesis/corridas/entrenamiento3_combinacion1')
df5=pd.read_csv('/home/felipe/Documentos/tesis/corridas/entrenamiento3_combinacion2')
df6=pd.read_csv('/home/felipe/Documentos/tesis/corridas/entrenamiento1_combinacion2') #
df7=pd.read_csv('/home/felipe/Documentos/tesis/corridas/entrenamiento2_combinacion1') #
df8=pd.read_csv('/home/felipe/Documentos/tesis/corridas/entrenamiento2_combinacion3') #
df9=pd.read_csv('/home/felipe/Documentos/tesis/corridas/entrenamiento1_combinacion3') #

plt.figure(figsize=(20,12))

dimensiones=df1['dimension']
cantidad_bien_train=df1['puntos_bien_train']
plt.plot(dimensiones,cantidad_bien_train, '-o',linewidth=1.2, label='combinacion 1 entrenamiento 1')#, label='puntos train')

dimensiones=df2['dimension']
cantidad_bien_train=df2['puntos_bien_train']
plt.plot(dimensiones,cantidad_bien_train, '-o',linewidth=1.2, label='combinacion 2 entrenamiento 2')#, label='puntos train')

dimensiones=df3['dimension']
cantidad_bien_train=df3['puntos_bien_train']
plt.plot(dimensiones,cantidad_bien_train, '-o',linewidth=1.2, label='combinacion 3 entrenamiento 3')#, label='puntos train')

dimensiones=df4['dimension']
cantidad_bien_train=df4['puntos_bien_train']
plt.plot(dimensiones,cantidad_bien_train, '-o',linewidth=1.2, label='combinacion 1 entrenamiento 3')#, label='puntos train')

dimensiones=df5['dimension']
cantidad_bien_train=df5['puntos_bien_train']
plt.plot(dimensiones,cantidad_bien_train, '-o',linewidth=1.2, label='combinacion 2 entrenamiento 3')#, label='puntos train')

dimensiones=df6['dimension']
cantidad_bien_train=df6['puntos_bien_train']
plt.plot(dimensiones,cantidad_bien_train, '-o',linewidth=1.2, label='combinacion 2 entrenamiento 1')#, label='puntos train')

dimensiones=df7['dimension']
cantidad_bien_train=df7['puntos_bien_train']
plt.plot(dimensiones,cantidad_bien_train, '-o',linewidth=1.2, label='combinacion 1 entrenamiento 2')#, label='puntos train')

dimensiones=df8['dimension']
cantidad_bien_train=df8['puntos_bien_train']
plt.plot(dimensiones,cantidad_bien_train, '-o',linewidth=1.2, label='combinacion 3 entrenamiento 2')#, label='puntos train')

dimensiones=df9['dimension']
cantidad_bien_train=df9['puntos_bien_train']
plt.plot(dimensiones,cantidad_bien_train, '-o',linewidth=1.2, label='combinacion 3 entrenamiento 1')#, label='puntos train')

plt.ylabel('Puntos bien clasif.')
plt.xlabel('Dimensiones')
#   plt.ylim(20,45)
plt.legend()