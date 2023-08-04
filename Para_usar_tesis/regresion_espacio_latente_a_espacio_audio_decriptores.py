#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 12:27:03 2022

@author: felipe
"""
import IPython
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import librosa as lib
from librosa import display
import seaborn as sns
#sns.set_theme
sns.set_theme(style="whitegrid")
from keras import models
from keras import layers
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import random
from tensorflow.keras.models import load_model
tf.compat.v1.enable_eager_execution()




rolloff=np.loadtxt('/home/felipe/Documents/tesis/audiodescriptores_cada_muestra/Rolloff.txt')
tiempos_ataque=np.loadtxt('/home/felipe/Documents/tesis/audiodescriptores_cada_muestra/Tiempos_ataque.txt')
centroides=np.loadtxt('/home/felipe/Documents/tesis/audiodescriptores_cada_muestra/Centroides.txt')
bandwidth=np.loadtxt('/home/felipe/Documents/tesis/audiodescriptores_cada_muestra/Bandwidth.txt')
flatness = np.loadtxt('/home/felipe/Documents/tesis/audiodescriptores_cada_muestra/Flatness.txt')
rms=np.loadtxt('/home/felipe/Documents/tesis/audiodescriptores_cada_muestra/rms.txt')
colores=np.repeat(('#000000','#be1e2d','#ffde17','#b2b2b2','#21409a'), 14)

#falta agregar el ultimo audioindcador al entrenamiento.

coordenadas_train=np.loadtxt('/home/felipe/Documents/tesis/audiodescriptores_cada_muestra/lista_coordenadas_train.txt')
coordenadas_val=np.loadtxt('/home/felipe/Documents/tesis/audiodescriptores_cada_muestra/lista_coordenadas_val.txt')
#%%%
angulo_1 = 20
angulo_2 = 70

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')

#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(coordenadas_train[:,0],coordenadas_train[:,1],coordenadas_train[:,2], marker='o',color=np.repeat(('#ffde17','#be1e2d','#21409a'), 14), s=70)
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_zlabel('Dim3')
ax.view_init(angulo_1, angulo_2)
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='o', color=np.repeat(('#000000','#be1e2d'), 14), s=70)
#ax.scatter(coordenadas_train[:,0],coordenadas_train[:,1],coordenadas_train[:,2], marker='o',color=np.repeat(('#ffde17','#b2b2b2','#21409a'), 14))
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_zlabel('Dim3')
ax.view_init(angulo_1, angulo_2)
plt.show()
#%% FIGURA PARA TESIS con alpha=1
angulo_1 = 20
angulo_2 = 70

lista_coordenadas=coordenadas_train
lista_coordenadas_val= coordenadas_val


fig = plt.figure(figsize=(15,7.5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(lista_coordenadas[0:14,0], lista_coordenadas[0:14,1], lista_coordenadas[0:14,2], c=np.repeat(('#ffde17'), 14), label='Variedad 1', alpha=1)
ax.scatter(lista_coordenadas[14:28,0], lista_coordenadas[14:28,1], lista_coordenadas[28:42,2], c=np.repeat(('#1A4A24'), 14), label='Variedad 2', alpha=1)
ax.scatter(lista_coordenadas[28:42,0], lista_coordenadas[28:42,1], lista_coordenadas[28:42,2], c=np.repeat(('#21409a'), 14), label='Variedad 3', alpha=1)
#ax.scatter(lista_coordenadas_val[:,0], lista_coordenadas_val[:,1], lista_coordenadas_val[:,2], c=lista_clases_val)
# ax.scatter(lista_coordenadas_val[:,0], lista_coordenadas_val[:,1], lista_coordenadas_val[:,2], c=lista_clases_val,marker="X")
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')
plt.legend()
plt.title('Training Classes')
ax.view_init(angulo_1, angulo_2)

x_limites = ax.get_xlim()
y_limites = ax.get_ylim()
z_limites = ax.get_zlim()

ax1 = fig.add_subplot(122, projection='3d')
ax1.scatter(lista_coordenadas_val[0:14,0], lista_coordenadas_val[0:14,1], lista_coordenadas_val[0:14,2], c=np.repeat(('#000000'), 14), label='Variedad 4', alpha=1)
ax1.scatter(lista_coordenadas_val[14:28,0], lista_coordenadas_val[14:28,1], lista_coordenadas_val[14:28,2], c=np.repeat(('#be1e2d'), 14), label='Variedad 5', alpha=1)
ax1.set_xlabel('Dim 1')
ax1.set_ylabel('Dim 2')
ax1.set_zlabel('Dim 3')
#ax1.set_xlim(x_limites)
#ax1.set_ylim(y_limites)
#ax1.set_zlim(z_limites)
plt.title('Validation Classes')
plt.legend()
ax1.view_init(angulo_1, angulo_2)
plt.tight_layout()
plt.show()

#%% FIGURA PARA TESIS ain alpha fijo
angulo_1 = 20
angulo_2 = 70

lista_coordenadas=coordenadas_train
lista_coordenadas_val= coordenadas_val


fig = plt.figure(figsize=(15,7.5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(lista_coordenadas[0:14,0], lista_coordenadas[0:14,1], lista_coordenadas[0:14,2], c=np.repeat(('#ffde17'), 14), label='Variedad 1')
ax.scatter(lista_coordenadas[14:28,0], lista_coordenadas[14:28,1], lista_coordenadas[28:42,2], c=np.repeat(('#1A4A24'), 14), label='Variedad 2')
ax.scatter(lista_coordenadas[28:42,0], lista_coordenadas[28:42,1], lista_coordenadas[28:42,2], c=np.repeat(('#21409a'), 14), label='Variedad 3')
#ax.scatter(lista_coordenadas_val[:,0], lista_coordenadas_val[:,1], lista_coordenadas_val[:,2], c=lista_clases_val)
# ax.scatter(lista_coordenadas_val[:,0], lista_coordenadas_val[:,1], lista_coordenadas_val[:,2], c=lista_clases_val,marker="X")
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')
plt.legend()
plt.title('Training Classes')
ax.view_init(angulo_1, angulo_2)

x_limites = ax.get_xlim()
y_limites = ax.get_ylim()
z_limites = ax.get_zlim()

ax1 = fig.add_subplot(122, projection='3d')
ax1.scatter(lista_coordenadas_val[0:14,0], lista_coordenadas_val[0:14,1], lista_coordenadas_val[0:14,2], c=np.repeat(('#000000'), 14), label='Variedad 4')
ax1.scatter(lista_coordenadas_val[14:28,0], lista_coordenadas_val[14:28,1], lista_coordenadas_val[14:28,2], c=np.repeat(('#be1e2d'), 14), label='Variedad 5')
ax1.set_xlabel('Dim 1')
ax1.set_ylabel('Dim 2')
ax1.set_zlabel('Dim 3')
#ax1.set_xlim(x_limites)
#ax1.set_ylim(y_limites)
#ax1.set_zlim(z_limites)
plt.title('Validation Classes')
plt.legend()
ax1.view_init(angulo_1, angulo_2)
plt.tight_layout()
plt.show()
#%%%CLUSTERING PARA LA TESIS
from sklearn.cluster import KMeans

lista_coordenadas=coordenadas_train
lista_coordenadas_val= coordenadas_val


kmeans_coordenadas = KMeans(n_clusters=3, random_state=0).fit(lista_coordenadas)
            #print('etiquetas de clustering',kmeans_coordenadas.labels_)
            #print('etiquetas de reales',lista_clases)

kmeans_coordenadas_val = KMeans(n_clusters=2, random_state=0).fit(lista_coordenadas_val)
            #print('etiquetas de clustering',kmeans_coordenadas_val.labels_)
            #print('etiquetas de reales',lista_clases_val)
            
angulo_1 = 20
angulo_2 = 70


 #armar bien los colores
kmeans_coordenadas.labels_colores_bien= np.where( kmeans_coordenadas.labels_ == 0 ,'#ffde17',  kmeans_coordenadas.labels_ )
kmeans_coordenadas.labels_colores_bien= np.where( kmeans_coordenadas.labels_ == 1, '#1A4A24'  ,  kmeans_coordenadas.labels_colores_bien)
kmeans_coordenadas.labels_colores_bien= np.where( kmeans_coordenadas.labels_ ==2 ,  '#21409a'  , kmeans_coordenadas.labels_colores_bien)

kmeans_coordenadas_val.labels_colores_bien= np.where( kmeans_coordenadas_val.labels_ == 0 ,'#000000',  kmeans_coordenadas_val.labels_ )
kmeans_coordenadas_val.labels_colores_bien= np.where( kmeans_coordenadas_val.labels_ == 1, '#be1e2d'  ,  kmeans_coordenadas_val.labels_colores_bien)



fig = plt.figure(figsize=(15,7.5))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(lista_coordenadas[:,0], lista_coordenadas[:,1], lista_coordenadas[:,2],c=kmeans_coordenadas.labels_colores_bien,  alpha=1)
ax.scatter(-1.43569   , -2.87443375,  3.95947375, c='#ffde17', alpha=1, label='cent 1', marker="X")
ax.scatter(-2.59198071, -4.11482857,  2.906485, c='#1A4A24', alpha=1, label='cent 2', marker="X")
ax.scatter(-1.933808  , -3.4955115 ,  3.6569025, c='#21409a', alpha=1, label='cent 3', marker="X")
#ax.scatter(lista_coordenadas[14:28,0], lista_coordenadas[14:28,1], lista_coordenadas[28:42,2], c=np.repeat(('#1A4A24'), 14), label='Variedad 2')
#ax.scatter(lista_coordenadas[28:42,0], lista_coordenadas[28:42,1], lista_coordenadas[28:42,2], c=np.repeat(('#21409a'), 14), label='Variedad 3')
#ax.scatter(lista_coordenadas_val[:,0], lista_coordenadas_val[:,1], lista_coordenadas_val[:,2], c=lista_clases_val)
# ax.scatter(lista_coordenadas_val[:,0], lista_coordenadas_val[:,1], lista_coordenadas_val[:,2], c=lista_clases_val,marker="X")
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')
plt.legend()
plt.title('Etiquetado vía clustering')
ax.view_init(angulo_1, angulo_2)

x_limites = ax.get_xlim()
y_limites = ax.get_ylim()
z_limites = ax.get_zlim()



ax = fig.add_subplot(122, projection='3d')
ax.scatter(lista_coordenadas[0:14,0], lista_coordenadas[0:14,1], lista_coordenadas[0:14,2], c=np.repeat(('#ffde17'), 14), label='Variedad 1', alpha=1)
ax.scatter(lista_coordenadas[14:28,0], lista_coordenadas[14:28,1], lista_coordenadas[28:42,2], c=np.repeat(('#1A4A24'), 14), label='Variedad 2', alpha=1)
ax.scatter(lista_coordenadas[28:42,0], lista_coordenadas[28:42,1], lista_coordenadas[28:42,2], c=np.repeat(('#21409a'), 14), label='Variedad 3', alpha=1)
#ax.scatter(lista_coordenadas_val[:,0], lista_coordenadas_val[:,1], lista_coordenadas_val[:,2], c=lista_clases_val)
# ax.scatter(lista_coordenadas_val[:,0], lista_coordenadas_val[:,1], lista_coordenadas_val[:,2], c=lista_clases_val,marker="X")
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')
plt.legend()
plt.title('Etiquetas reales')
ax.view_init(angulo_1, angulo_2)

x_limites = ax.get_xlim()
y_limites = ax.get_ylim()
z_limites = ax.get_zlim()




#%%%
ax1 = fig.add_subplot(122, projection='3d')
ax1.scatter(lista_coordenadas_val[:,0], lista_coordenadas_val[:,1], lista_coordenadas_val[:,2], c=kmeans_coordenadas_val.labels_colores_bien , label='Variedad 4', alpha=1)
#ax1.scatter(lista_coordenadas_val[14:28,0], lista_coordenadas_val[14:28,1], lista_coordenadas_val[14:28,2], c=np.repeat(('#be1e2d'), 14), label='Variedad 5')
ax1.set_xlabel('Dim 1')
ax1.set_ylabel('Dim 2')
ax1.set_zlabel('Dim 3')
#ax1.set_xlim(x_limites)
#ax1.set_ylim(y_limites)
#ax1.set_zlim(z_limites)
plt.title('Validation Classes')
plt.legend()
ax1.view_init(angulo_1, angulo_2)
plt.tight_layout()
plt.show()

#%%% PREPARACION PARA ENTRENAMIENTO

#Separemos los de validacion y train

rolloff_train=rolloff[0:42]
rolloff_val=rolloff[42:70]

centroides_train=centroides[0:42]
centroides_val=centroides[42:70]

tiempos_ataque_train=tiempos_ataque[0:42]
tiempos_ataque_val=tiempos_ataque[42:70]

bandwidth_train=bandwidth[0:42]
bandwidth_val=bandwidth[42:70]

flatness_train=flatness[0:42]
flatness_val=flatness[42:70]

rms_train=rms[0:42]
rms_val=rms[42:70]

#%%%Normalicemos cada una por su maximo

rolloff_train=rolloff[0:42]/np.max(rolloff)
rolloff_val=rolloff[42:70]/np.max(rolloff)

centroides_train=centroides[0:42]/np.max(centroides)
centroides_val=centroides[42:70]/np.max(centroides)

tiempos_ataque_train=tiempos_ataque[0:42]/np.max(tiempos_ataque)
tiempos_ataque_val=tiempos_ataque[42:70]/np.max(tiempos_ataque)

bandwidth_train=bandwidth[0:42]/np.max(bandwidth)
bandwidth_val=bandwidth[42:70]/np.max(bandwidth)

flatness_train=flatness[0:42]/np.max(flatness)
flatness_val=flatness[42:70]/np.max(flatness)

rms_train=rms[0:42]/np.max(rms)
rms_val=rms[42:70]/np.max(rms)
#%%%Normalicemos por el maximo total

rolloff_train=rolloff[0:42]/np.max(rolloff)
rolloff_val=rolloff[42:70]/np.max(rolloff)

centroides_train=centroides[0:42]/np.max(rolloff)
centroides_val=centroides[42:70]/np.max(rolloff)

tiempos_ataque_train=tiempos_ataque[0:42]/np.max(rolloff)
tiempos_ataque_val=tiempos_ataque[42:70]/np.max(rolloff)

bandwidth_train=bandwidth[0:42]/np.max(rolloff)
bandwidth_val=bandwidth[42:70]/np.max(rolloff)

flatness_train=flatness[0:42]/np.max(rolloff)
flatness_val=flatness[42:70]/np.max(rolloff)
#%%%

audio_descriptores_train = np.column_stack((rolloff_train,centroides_train,tiempos_ataque_train))#, bandwidth_train))#, flatness_train))#, rms_train))
audio_descriptores_val = np.column_stack((rolloff_val,centroides_val,tiempos_ataque_val))#, bandwidth_val))#, flatness_val))#, rms_val))
np.savetxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/Audio_descriptores_val.txt',audio_descriptores_val,fmt = '%10.5f')
train_data = coordenadas_train
train_targets = audio_descriptores_train

#%% Generamos datos aleatorios
from random import gauss

def generar_ruido(vector, porcentaje_error , cantidad_puntos_ruidosos):
    vector_con_ruido=[]
    for i in np.arange(0,len(vector),1):
        valor=vector[i]
        valor_ruidoso=[random.gauss(valor,(porcentaje_error * valor)/100) for j in range(cantidad_puntos_ruidosos)]
        vector_con_ruido=np.concatenate((vector_con_ruido,valor_ruidoso))
    return vector_con_ruido   

porcentaje_error=1.5
numero=100

#audio_descriptores
centroides_train_ampliado=generar_ruido(centroides_train,porcentaje_error,numero)
rolloff_train_ampliado=generar_ruido(rolloff_train,porcentaje_error,numero)
bandwidth_train_ampliado=generar_ruido(bandwidth_train,porcentaje_error,numero)
flatness_train_ampliado=generar_ruido(flatness_train,porcentaje_error,numero)
tiempos_ataque_train_ampliado=generar_ruido(tiempos_ataque_train,porcentaje_error,numero)
rms_train_ampliado=generar_ruido(rms_train, porcentaje_error, numero)

audio_descriptores_train = np.column_stack((rolloff_train_ampliado,centroides_train_ampliado,tiempos_ataque_train_ampliado))#, bandwidth_train_ampliado, flatness_train_ampliado, rms_train_ampliado))
train_targets = audio_descriptores_train

#coordenadas
coord1_ampliada=generar_ruido(coordenadas_train[:,0],porcentaje_error,numero)
coord2_ampliada=generar_ruido(coordenadas_train[:,1],porcentaje_error,numero)
coord3_ampliada=generar_ruido(coordenadas_train[:,2],porcentaje_error,numero)

coordenadas_train= np.column_stack((coord1_ampliada,coord2_ampliada,coord3_ampliada))
train_data  = coordenadas_train

#%%% Entrenamiento desde espacio latente hacia audio descriptores

#train_data = coordenadas_train
#train_targets = audio_descriptores_train

#print(len(train_data))

#from sklearn.utils import shuffle #vamos a aleatorizar los numeros de entrada

#train_data, train_targets = shuffle(train_data, train_targets, random_state=0)

delays_medidos=np.zeros(train_data.shape[1])
activador='selu'
#entrenamos la red
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(6, activation=activador,
                           input_shape=(train_data.shape[1],)))
    
    #model.add(layers.Dense(16, activation=activador))
    #model.add(layers.Dense(32, activation=activador))
    #model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dense(64, activation=activador))
    #model.add(layers.Dense(128, activation=activador))
    #model.add(layers.Dense(64, activation=activador))
    #model.add(layers.Dense(32, activation='relu'))
    #model.add(layers.Dense(16, activation='relu'))
    #model.add(layers.Dense(12, activation='relu'))
    model.add(layers.Dense(12, activation=activador))
    #model.add(layers.Dense(24, activation=activador))
    model.add(layers.Dense(12, activation=activador))
    model.add(layers.Dense(3))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 3 #4
num_val_samples = len(train_data) // k
num_epochs =60
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],
         train_data[(i + 1) * num_val_samples:]],
         axis=0)
    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],
         train_targets[(i + 1) * num_val_samples:]],
         axis=0)
    model = build_model()
    history=model.fit(partial_train_data, partial_train_targets,
                      validation_data=(val_data, val_targets),
                      epochs=num_epochs, batch_size=1) #verbose=0
    
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    mae_history = history.history['mae']
    all_mae_histories.append(mae_history)

model.save('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/latente_a_audiodescrip_3d_prueba1.h5') #guarden el modelo en su Drive personal
average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history,'.')
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
#plt.ylim(np.min(average_mae_history)-1,np.min(average_mae_history)+1)
plt.hlines(np.min(average_mae_history),0,num_epochs,colors='r', linestyles='dashed')
plt.show()
print('El MAE minimo es: ',np.min(average_mae_history))

#%%%

from tensorflow.keras.models import load_model
modelo = load_model('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/latente_a_audiodescrip_3d_prueba1.h5')


#np.set_printoptions(precision=3, suppress=True) #para quitar la notación científica

# definir el vector de los delays
selected_data=coordenadas_val
xyz=modelo.predict(selected_data) #pasamos los datos por la red
print('Posiciones predichas')
print('delays',selected_data.shape)
print(xyz)

pos_reales=audio_descriptores_val
print()
print('Posiciones reales')
print('pos',pos_reales.shape)
print(pos_reales)

#veamos cuan distinto da

B=np.abs(xyz-pos_reales)

#error_train=np.sqrt( (np.mean(A[:,0]))**2 + (np.mean(A[:,1]))**2 + (np.mean(A[:,2]))**2 )
error_selected=np.sqrt( (np.mean(B[:,0]))**2 + (np.mean(B[:,1]))**2 + (np.mean(B[:,2]))**2  )

#print('El error de entrenamiento es de',error_train)
print('El error en los datos de validacion es de', error_selected )

#Veamos los histogramas

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
#ax[0,0].hist(A[:,0])
#ax[0,0].set_title('Error de train Dim 1')

#ax[0,1].hist(A[:,1])
#ax[0,1].set_title('Error de train Dim 2')

#ax[0,2].hist(A[:,2])
#ax[0,2].set_title('Error de train Dim 3')


ax[0].hist(B[:,0])
ax[0].set_title('Error de val Dim 1')

ax[1].hist(B[:,1])
ax[1].set_title('Error de val Dim 2')

ax[2].hist(B[:,2])
ax[2].set_title('Error de val Dim 3')

#ax[1,1].hist(B[:,3])
#ax[1,1].set_title('Error de val Dim 4')

#ax[0,2].hist(B[:,4])
#ax[0,2].set_title('Error de val Dim 5')

#ax[1,2].hist(B[:,5])
#ax[1,2].set_title('Error de val Dim 6')



plt.show()


#%%% Hay dos dimensiones que tienen un error muy bajo: la 5 y la 3, o sea tiempo de ataque y flatness. Grafiquemos

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(np.mean(xyz[0:14,2]),np.mean(xyz[0:14,4]), marker='o',color='#000000', s=70,label='Prediccion 1')
ax.scatter(np.mean(xyz[14:28,2]),np.mean(xyz[14:28,4]), marker='o',color='#be1e2d', s=70,label='Prediccion 2')

ax.scatter(np.mean(pos_reales[0:14,2]),np.mean(pos_reales[0:14,4]), marker='o',color='#ffde17', s=70, label='Dato 1')
ax.scatter(np.mean(pos_reales[14:28,2]),np.mean(pos_reales[14:28,4]), marker='o',color='#21409a', s=70, label='Dato 2')


ax.set_xlabel('Tiempos de Ataque')
ax.set_ylabel('Flatness')
plt.title('Comparacion Prediccion vs Datos')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(xyz[0:14,2],xyz[0:14,4], marker='o',color='#000000', s=70,label='Prediccion 1')
ax.scatter(xyz[14:28,2],xyz[14:28,4], marker='o',color='#be1e2d', s=70,label='Prediccion 2')

ax.scatter(pos_reales[0:14,2],pos_reales[0:14,4], marker='o',color='#ffde17', s=70, label='Dato 1')
ax.scatter(pos_reales[14:28,2],pos_reales[14:28,4], marker='o',color='#21409a', s=70, label='Dato 2')


ax.set_xlabel('Tiempos de Ataque')
ax.set_ylabel('Flatness')
plt.title('Comparacion Prediccion vs Datos')
plt.legend()
plt.show()
#%%%
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2], marker='o',color=np.repeat(('#be1e2d','#21409a'), 14), s=70)
ax.set_xlabel('Rolloff Espectral')
ax.set_ylabel('Tiempos de Ataque')
ax.set_zlabel('Centroides Espectrales')
plt.title('posiciones predichas')
plt.show()


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(pos_reales[:,0],pos_reales[:,1],pos_reales[:,2], marker='o',color=np.repeat(('#be1e2d','#21409a'), 14), s=70)
ax.set_xlabel('Rolloff Espectral')
ax.set_ylabel('Tiempos de Ataque')
ax.set_zlabel('Centroides Espectrales')
plt.title('posiciones reales')
plt.show()

#%%% medias

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(np.mean(xyz[0:14,0]),np.mean(xyz[0:14,1]),np.mean(xyz[0:14,2]), marker='o',color='#000000', s=70,label='Prediccion 1')
ax.scatter(np.mean(xyz[14:28,0]),np.mean(xyz[14:28,1]),np.mean(xyz[14:28,2]), marker='o',color='#be1e2d', s=70,label='Prediccion 2')

ax.scatter(np.mean(pos_reales[0:14,0]),np.mean(pos_reales[0:14,1]),np.mean(pos_reales[0:14,2]), marker='x',color='#000000', s=70, label='Dato 1')

ax.scatter(np.mean(pos_reales[14:28,0]),np.mean(pos_reales[14:28,1]),np.mean(pos_reales[14:28,2]), marker='x',color='#be1e2d', s=70, label='Dato 2')


ax.set_xlabel('Rolloff Espectral')
ax.set_ylabel('Tiempos de Ataque')
ax.set_zlabel('Centroides Espectrales')
plt.title('Comparacion Prediccion vs Datos Validacion 1')
plt.legend()
plt.show()

xyz2=modelo.predict(coordenadas_train)
pos_reales2=train_targets

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(np.mean(xyz2[0:14,0]),np.mean(xyz2[0:14,1]),np.mean(xyz2[0:14,2]), marker='o',color='#000000', s=70,label='Prediccion 1')
ax.scatter(np.mean(xyz2[14:28,0]),np.mean(xyz2[14:28,1]),np.mean(xyz2[14:28,2]), marker='o',color='#be1e2d', s=70,label='Prediccion 2')

ax.scatter(np.mean(pos_reales2[0:14,0]),np.mean(pos_reales2[0:14,1]),np.mean(pos_reales2[0:14,2]), marker='x',color='#000000', s=70, label='Dato 1')

ax.scatter(np.mean(pos_reales2[14:28,0]),np.mean(pos_reales2[14:28,1]),np.mean(pos_reales2[14:28,2]), marker='x',color='#be1e2d', s=70, label='Dato 2')


ax.set_xlabel('Rolloff Espectral')
ax.set_ylabel('Tiempos de Ataque')
ax.set_zlabel('Centroides Espectrales')
plt.title('Comparacion Prediccion vs Datos Entrenamiento')
plt.legend()
plt.show()
#%%% FIGURAS TESIS


#todo junto
fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(projection='3d')
angulo_1 = 20
angulo_2 = 70
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(np.mean(xyz[0:14,0]),np.mean(xyz[0:14,1]),np.mean(xyz[0:14,2]), marker='o',color='#000000', s=70,label='Prediccion 1')
ax.scatter(xyz[0:14,0],xyz[0:14,1],xyz[0:14,2], marker='o',color='#000000',alpha=0.2,  s=70)
ax.scatter(np.mean(xyz[14:28,0]),np.mean(xyz[14:28,1]),np.mean(xyz[14:28,2]), marker='o',color='#be1e2d', s=70,label='Prediccion 2')
ax.scatter(xyz[14:28,0],xyz[14:28,1],xyz[14:28,2], marker='o',color='#be1e2d',alpha=0.2, s=70)

ax.scatter(np.mean(pos_reales[0:14,0]),np.mean(pos_reales[0:14,1]),np.mean(pos_reales[0:14,2]), marker='x',color='#000000', s=70, label='Dato 1')
ax.scatter(pos_reales[0:14,0],pos_reales[0:14,1],pos_reales[0:14,2], marker='x',color='#000000',alpha=0.2, s=70 )

ax.scatter(np.mean(pos_reales[14:28,0]),np.mean(pos_reales[14:28,1]),np.mean(pos_reales[14:28,2]), marker='x',color='#be1e2d', s=70, label='Dato 2')
ax.scatter(pos_reales[14:28,0],pos_reales[14:28,1],pos_reales[14:28,2], marker='x',color='#be1e2d',alpha=0.2, s=70)
ax.view_init(angulo_1, angulo_2)
ax.set_xlabel('RMS[u.a.]')
ax.set_ylabel('Tiempos de Ataque[u.a.]')
ax.set_zlabel('Centroides Espectrales[u.a.]')
plt.title('Comparacion Prediccion vs Datos Validacion ')
plt.legend()
plt.show()


#por separado
#primer variedad
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(np.mean(xyz[0:14,0]),np.mean(xyz[0:14,1]),np.mean(xyz[0:14,2]), marker='o',color='#000000', s=70,label='Prediccion 1')
ax.scatter(xyz[0:14,0],xyz[0:14,1],xyz[0:14,2], marker='o',color='#000000',alpha=0.2,  s=70)
ax.scatter(np.mean(pos_reales[0:14,0]),np.mean(pos_reales[0:14,1]),np.mean(pos_reales[0:14,2]), marker='x',color='#000000', s=70, label='Dato 1')
ax.scatter(pos_reales[0:14,0],pos_reales[0:14,1],pos_reales[0:14,2], marker='x',color='#000000',alpha=0.2, s=70)

ax.set_xlabel('RMS[u.a.]')
ax.set_ylabel('Tiempos de Ataque[u.a.]')
ax.set_zlabel('Centroides Espectrales[u.a.]')
ax.view_init(angulo_1, angulo_2)
plt.title('Comparacion Prediccion vs Datos Variedad (Variedad 1: Trompeta) ')
plt.legend()
plt.show()
#primer variedad
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(np.mean(xyz[14:28,0]),np.mean(xyz[14:28,1]),np.mean(xyz[14:28,2]), marker='o',color='#be1e2d', s=70,label='Prediccion 2')
ax.scatter(xyz[14:28,0],xyz[14:28,1],xyz[14:28,2], marker='o',color='#be1e2d',alpha=0.2, s=70)
ax.scatter(np.mean(pos_reales[14:28,0]),np.mean(pos_reales[14:28,1]),np.mean(pos_reales[14:28,2]), marker='x',color='#be1e2d', s=70, label='Dato 2')
ax.scatter(pos_reales[14:28,0],pos_reales[14:28,1],pos_reales[14:28,2], marker='x',color='#be1e2d',alpha=0.2, s=70)
ax.set_xlabel('RMS[u.a.]')
ax.set_ylabel('Tiempo Ataque[u.a.]')
ax.set_zlabel('Cent. Espec. [u.a.]')
ax.view_init(angulo_1, angulo_2)
plt.title('Comparacion Prediccion vs Datos Validacion (Variedad 2: Violin)')
plt.legend()
plt.show()

#las dos juntas en una
fig = plt.figure(figsize=(20,10))
ax = fig.add_subplot(121, projection='3d')
ax.scatter(np.mean(xyz[0:14,0]),np.mean(xyz[0:14,1]),np.mean(xyz[0:14,2]), marker='o',color='#000000', s=70,label='Prediccion 1')
ax.scatter(xyz[0:14,0],xyz[0:14,1],xyz[0:14,2], marker='o',color='#000000',alpha=0.2,  s=70)
ax.scatter(np.mean(pos_reales[0:14,0]),np.mean(pos_reales[0:14,1]),np.mean(pos_reales[0:14,2]), marker='x',color='#000000', s=70, label='Dato 1')
ax.scatter(pos_reales[0:14,0],pos_reales[0:14,1],pos_reales[0:14,2], marker='x',color='#000000',alpha=0.2, s=70)
ax.set_xlabel('RMS[u.a.]')
ax.set_ylabel('Tiempos de Ataque[u.a.]')
ax.set_zlabel('Centroides Espectrales[u.a.]')
ax.view_init(angulo_1, angulo_2)
plt.title('Comparacion Prediccion vs Datos Validacion (Variedad 1: Trompeta)')
plt.legend()



ax1 = fig.add_subplot(122, projection='3d')
ax1.scatter(np.mean(xyz[14:28,0]),np.mean(xyz[14:28,1]),np.mean(xyz[14:28,2]), marker='o',color='#be1e2d', s=70,label='Prediccion 2')
ax1.scatter(xyz[14:28,0],xyz[14:28,1],xyz[14:28,2], marker='o',color='#be1e2d',alpha=0.2, s=70)
ax1.scatter(np.mean(pos_reales[14:28,0]),np.mean(pos_reales[14:28,1]),np.mean(pos_reales[14:28,2]), marker='x',color='#be1e2d', s=70, label='Dato 2')
ax1.scatter(pos_reales[14:28,0],pos_reales[14:28,1],pos_reales[14:28,2], marker='x',color='#be1e2d',alpha=0.2, s=70)
ax1.set_xlabel('RMS[u.a.]')
ax1.set_ylabel('Tiempo Ataque[u.a.]')
ax1.set_zlabel('Cent. Espec. [u.a.]')
ax1.view_init(angulo_1, angulo_2)
plt.title('Comparacion Prediccion vs Datos Validacion (Variedad 2: Violin)')
plt.legend()
plt.show()


#%%%

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot()
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(np.mean(xyz[0:14,3]),np.mean(xyz[0:14,4]),np.mean(xyz[0:14,5]), marker='o',color='#000000', s=70,label='Prediccion 1')
ax.scatter(np.mean(xyz[14:28,3]),np.mean(xyz[14:28,4]),np.mean(xyz[14:28,5]), marker='o',color='#be1e2d', s=70,label='Prediccion 2')

ax.scatter(np.mean(pos_reales[0:14,3]),np.mean(pos_reales[0:14,4]),np.mean(pos_reales[0:14,5]), marker='x',color='#000000', s=70, label='Dato 1')

ax.scatter(np.mean(pos_reales[14:28,3]),np.mean(pos_reales[14:28,4]),np.mean(pos_reales[14:28,5]), marker='x',color='#be1e2d', s=70, label='Dato 2')


ax.set_xlabel('Spectral bandwidth')
ax.set_ylabel('Spectral flatness')
#ax.set_zlabel('RMS')
plt.title('Comparacion Prediccion vs Datos Validacion 2')
plt.legend()
plt.show()
#%%% en 4d

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#img1 = ax.scatter(np.mean(pos_reales[0:14,0]),np.mean(pos_reales[0:14,1]),np.mean(pos_reales[0:14,2]),
 #                 c=np.mean(pos_reales[0:14,3]), cmap=plt.hot(), marker='o',s=70, label='Dato 1')

img1 = ax.scatter(np.mean(pos_reales[14:28,0]),np.mean(pos_reales[14:28,1]),np.mean(pos_reales[14:28,2]),
                                   c=np.mean(pos_reales[14:28,3]), cmap=plt.hot(), marker='o',s=70, label='Dato 2')#,color='#ffde17')c=np.mean(pos_reales[0:14,3]), cmap=plt.hot(), marker='o',s=70, label='Dato 1')#,color='#ffde17')

#img3=ax.scatter(np.mean(xyz[0:14,0]),np.mean(xyz[0:14,1]),np.mean(xyz[0:14,2]),
 #                 c=np.mean(xyz[0:14,3]), cmap=plt.hot(), marker='o',s=70, label='Prediccion 1')

img3=ax.scatter(np.mean(xyz[14:28,0]),np.mean(xyz[14:28,1]),np.mean(xyz[14:28,2]),
                  c=np.mean(xyz[14:28,3]), cmap=plt.hot(), marker='o',s=70, label='Prediccion 2')

fig.colorbar(img1)
plt.title('Comparacion Prediccion vs Datos')
ax.set_xlabel('Rolloff Espectral')
ax.set_ylabel('Tiempos de Ataque')
ax.set_zlabel('Centroides Espectrales')
plt.legend()
plt.show()
plt.show()
#%%% Sitematicemos el error
#A=coordenadas_train-modelo.predict(train_data)
B=pos_reales-xyz

#error_train=np.sqrt( (np.mean(A[:,0]))**2 + (np.mean(A[:,1]))**2 + (np.mean(A[:,2]))**2 )
error_selected=np.sqrt( (np.mean(B[:,0]))**2 + (np.mean(B[:,1]))**2 + (np.mean(B[:,2]))**2 )

#print('El error de entrenamiento es de',error_train)
print('El error en los datos de validacion es de', error_selected )

#Veamos los histogramas

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
#ax[0,0].hist(A[:,0])
#ax[0,0].set_title('Error de train Dim 1')

#ax[0,1].hist(A[:,1])
#ax[0,1].set_title('Error de train Dim 2')

#ax[0,2].hist(A[:,2])
#ax[0,2].set_title('Error de train Dim 3')


ax[0,0].hist(B[:,0])
ax[0,0].set_title('Error de val Dim 1')

ax[0,1].hist(B[:,1])
ax[0,1].set_title('Error de val Dim 2')

ax[1,0].hist(B[:,2])
ax[1,0].set_title('Error de val Dim 3')

ax[1,1].hist(B[:,3])
ax[1,1].set_title('Error de val Dim 4')

ax[0,2].hist(B[:,4])
ax[0,2].set_title('Error de val Dim 5')

ax[1,2].hist(B[:,5])
ax[1,2].set_title('Error de val Dim 6')



plt.show()
