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
sns.set_theme
from keras import models
from keras import layers
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import tensorflow as tf
import math
import random
from tensorflow.keras.models import load_model
tf.compat.v1.enable_eager_execution()
import seaborn as sns
sns.set_theme(style="ticks")



rolloff=np.loadtxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/Rolloff.txt')
tiempos_ataque=np.loadtxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/Tiempos_ataque.txt')
centroides=np.loadtxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/Centroides.txt')
colores=np.repeat(('#000000','#be1e2d','#ffde17','#b2b2b2','#21409a'), 14)

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(rolloff,tiempos_ataque,centroides, marker='^', color=colores)
ax.set_xlabel('Rolloff Espectral')
ax.set_ylabel('Tiempos de Ataque')
ax.set_zlabel('Centroides Espectrales')
plt.show()

coordenadas_train=np.loadtxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/lista_coordenadas_train.txt')
coordenadas_val=np.loadtxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/lista_coordenadas_val.txt')
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(coordenadas_train[:,0],coordenadas_train[:,1],coordenadas_train[:,2], marker='o',color=np.repeat(('#ffde17','#be1e2d','#21409a'), 14), s=70)
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_zlabel('Dim3')
plt.show()

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14), s=70)
#ax.scatter(coordenadas_train[:,0],coordenadas_train[:,1],coordenadas_train[:,2], marker='o',color=np.repeat(('#ffde17','#b2b2b2','#21409a'), 14))
ax.set_xlabel('Dim1')
ax.set_ylabel('Dim2')
ax.set_zlabel('Dim3')
plt.show()

#%%% PREPARACION PARA ENTRENAMIENTO

#Separemos los de validacion y train

rolloff_train=rolloff[0:42]
rolloff_val=rolloff[42:70]

centroides_train=centroides[0:42]
centroides_val=centroides[42:70]

tiempos_ataque_train=tiempos_ataque[0:42]
tiempos_ataque_val=tiempos_ataque[42:70]

audio_descriptores_train = np.column_stack((rolloff_train,centroides_train,tiempos_ataque_train))
audio_descriptores_val = np.column_stack((rolloff_val,centroides_val,tiempos_ataque_val))
#%%% Entrenamiento desde espacio latente hacia audio descriptores

train_data = audio_descriptores_train
train_targets = coordenadas_train

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
    
    #model.add(layers.Dense(16, activation='relu'))
    #model.add(layers.Dense(32, activation='relu'))
    #model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(128, activation='relu'))
    #model.add(layers.Dense(64, activation='relu'))
    #model.add(layers.Dense(32, activation='relu'))
    #model.add(layers.Dense(16, activation='relu'))
    #model.add(layers.Dense(12, activation='relu'))

    #model.add(layers.Dense(12, activation=activador))
    model.add(layers.Dense(6, activation=activador))
    model.add(layers.Dense(3))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model

k = 3 #4
num_val_samples = len(train_data) // k
num_epochs = 100
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

model.save('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/audiodescrip_a_latente_3d_prueba1.h5') #guarden el modelo en su Drive personal
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
modelo = load_model('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/audiodescrip_a_latente_3d_prueba1.h5')


#np.set_printoptions(precision=3, suppress=True) #para quitar la notación científica

# definir el vector de los delays
selected_data=audio_descriptores_val
xyz=modelo.predict(selected_data) #pasamos los datos por la red
print('Posiciones predecidas')
print('delays',selected_data.shape)
print(xyz)

pos_reales=coordenadas_val
print()
print('Posiciones reales')
print('pos',pos_reales.shape)
print(pos_reales)


fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2], marker='o',color=np.repeat(('#be1e2d','#21409a'), 14), s=70, label='predichas')
ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')
plt.title('posiciones predichas')
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(pos_reales[:,0],pos_reales[:,1],pos_reales[:,2], marker='o',color=np.repeat(('#b2b2b2','#000000'), 14), s=70, label='reales')
plt.legend()
plt.title('posiciones reales')
plt.show()
#%%% medias

fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(projection='3d')
#ax.scatter(coordenadas_val[:,0],coordenadas_val[:,1],coordenadas_val[:,2], marker='x', color=np.repeat(('#000000','#be1e2d'), 14))
ax.scatter(np.mean(xyz[0:14,0]),np.mean(xyz[0:14,1]),np.mean(xyz[0:14,2]), marker='o',color='#000000', s=70,label='Prediccion 1')
ax.scatter(np.mean(xyz[14:28,0]),np.mean(xyz[14:28,1]),np.mean(xyz[14:28,2]), marker='o',color='#be1e2d', s=70,label='Prediccion 2')

ax.scatter(np.mean(pos_reales[0:14,0]),np.mean(pos_reales[0:14,1]),np.mean(pos_reales[0:14,2]), marker='o',color='#ffde17', s=70, label='Dato 1')
ax.scatter(np.mean(pos_reales[14:28,0]),np.mean(pos_reales[14:28,1]),np.mean(pos_reales[14:28,2]), marker='o',color='#21409a', s=70, label='Dato 2')

ax.set_xlabel('Dim 1')
ax.set_ylabel('Dim 2')
ax.set_zlabel('Dim 3')
plt.title('Comparacion Prediccion vs Datos')
plt.legend()
plt.show()

#%%% Sitematicemos el error
A=coordenadas_train-modelo.predict(train_data)
B=pos_reales-xyz

error_train=np.sqrt( (np.mean(A[:,0]))**2 + (np.mean(A[:,1]))**2 + (np.mean(A[:,2]))**2 )
error_selected=np.sqrt( (np.mean(B[:,0]))**2 + (np.mean(B[:,1]))**2 + (np.mean(B[:,2]))**2 )

print('El error de entrenamiento es de',error_train)
print('El error en los datos de validacion es de', error_selected )

#Veamos los histogramas

fig, ax = plt.subplots(2, 3, figsize=(15, 10))
ax[0,0].hist(A[:,0])
ax[0,0].set_title('Error de train Dim 1')

ax[0,1].hist(A[:,1])
ax[0,1].set_title('Error de train Dim 2')

ax[0,2].hist(A[:,2])
ax[0,2].set_title('Error de train Dim 3')


ax[1,0].hist(B[:,0])
ax[1,0].set_title('Error de val Dim 1')

ax[1,1].hist(B[:,1])
ax[1,1].set_title('Error de val Dim 2')

ax[1,2].hist(B[:,2])
ax[1,2].set_title('Error de val Dim 3')

plt.show()

sns.violinplot(data=A[:,2],split=True, inner="quart", linewidth=1,
               palette={"Yes": "b", "No": ".85"})
sns.despine(left=True)