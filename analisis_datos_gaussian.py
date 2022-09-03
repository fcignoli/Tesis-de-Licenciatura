#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 11:29:45 2022

@author: felipe
"""

import os, sys 
import matplotlib.pyplot as plt, numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import ListedColormap
#import seaborn as sns
#sns.set_theme()
#sns.reset_orig()
from PIL import Image

import tensorflow as tf
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, merge  
from keras.layers import InputLayer, UpSampling2D, Concatenate, Reshape, Embedding, dot
from keras.models import Model, Sequential, load_model
from keras.regularizers import l2
from keras import backend as K
from tensorflow.keras.optimizers import SGD, Adam
from keras.losses import binary_crossentropy
# import keras_utils
from keras.utils.vis_utils import plot_model

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

from scipy import spatial
from scipy.spatial import distance

import mpl_toolkits.mplot3d  # noqa: F401
#from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import random
import seaborn as sns
sns.set_theme(style="darkgrid")
#from google.colab import drive
#drive.mount('/content/drive')
###########################################################################################################lases_train_tod
ubicacion_del_archivo='/home/felipe/Documentos/tesis/corridas/entrenamiento1_combinacion3'

cantidad_bien_train=[]
cantidad_bien_val=[]
mediana_cantidad_bien_train=[]
mediana_cantidad_bien_val=[]
desvio_cantidad_bien_train=[]
desvio_cantidad_bien_val=[]
dimensiones=[3,4,5,6,7,8,9,10,12,15,17,20,25,30,40,50]

for k in dimensiones:
    clases_train_todos=[]
    clases_val_todos=[]
    cantidad_true_train=[]
    #cantidad_false_train=[]
    cantidad_true_val=[]
    #cantidad_false_val=[0]
    num = 20
    for _ in range(num):
        
            salida= k
            pesos='/home/felipe/Documentos/tesis/modelo' + str(k) +'d_tamaños_ajustados.h5'
            
            
            base_dir = '/home/felipe/Documentos/tesis/espectogramas_entrenamiento'
            train_test_split = 0.625
            no_of_files_in_each_class = 14 # tomo una muestra de N imagenes por clase
            # 50 train; 30 test
            
            #Read all the folders in the directory
            folder_list = os.listdir(base_dir)
            folder_list.sort()
            #print(len(folder_list), "clases para construir el modelo")
            
            #Declare training array
            cat_list = []
            x = []
            y = []
            y_label = 0
            
            for folder_name in folder_list:
                files_list = os.listdir(os.path.join(base_dir, folder_name)) #una lista con los archivos de ese folder
                random.shuffle(files_list)  #randomizo
                temp=[] #va a hacer una lista de temp para cada folder
                for file_name in files_list[:no_of_files_in_each_class]:
                    temp.append(len(x)) #en cada elemento del folder, agrega a temp
                    x.append(np.asarray(Image.open(os.path.join(base_dir, folder_name, file_name)).convert('L').resize((300, 150))).reshape(300,150,1)) #¿por que se les cambia el tamaño?
                    y.append(y_label)
                y_label+=1
                cat_list.append(np.asarray(temp))
            
            cat_list = np.asarray(cat_list)
            x = np.asarray(x)/255.0
            y = np.asarray(y)
            #print('X, Y shape',x.shape, y.shape, cat_list.shape)        
            
            train_size = int(len(folder_list)*train_test_split)
            test_size = len(folder_list) - train_size
            #print(train_size, 'clases para el entrenamiento, y', test_size, ' clases para validación')
            
            train_files = train_size * no_of_files_in_each_class
            
            #Training Split
            x_train = x[:train_files]
            y_train = y[:train_files]
            cat_train = cat_list[:train_size]
            
            #Validation Split
            x_val = x[train_files:]
            y_val = y[train_files:]
            cat_test = cat_list[train_size:]
            
            #print('X&Y shape de los datos de entrenamiento :',x_train.shape, 'y', y_train.shape, cat_train.shape)
            #print('X&Y shape de los datos de validación :' , x_val.shape, 'y', y_val.shape, cat_test.shape)
            
            def get_batch(batch_size):    
                temp_x = x_train
                temp_cat_list = cat_train
                start=0
                end=train_size
                batch_x=[]
                    
                batch_y = np.zeros(batch_size)
                batch_y[int(batch_size/2):] = 1
                np.random.shuffle(batch_y)
                # armamos un vector de batch_size de ceros y unos, mezclados. TAMANOS
                class_list = np.random.randint(start, end, batch_size) 
                batch_x.append(np.zeros((batch_size, 300, 150,1)))
                batch_x.append(np.zeros((batch_size, 300, 150,1)))
            
                for i in range(0, batch_size):        
                    batch_x[0][i] = temp_x[np.random.choice(temp_cat_list[class_list[i]])]  
                    # If train_y has 0 pick from the same class, else pick from any other class
                    if batch_y[i]==0:
                        batch_x[1][i] = temp_x[np.random.choice(temp_cat_list[class_list[i]])]
                        batch_y[i] = 1 #agregado en esta nueva versión de Gonza/Gabo
                    else:
                        temp_list = np.append(temp_cat_list[:class_list[i]].flatten(), temp_cat_list[class_list[i]+1:].flatten())
                        # print(temp_list)
                        # print('1:',batch_x[0][i].shape)
                        # print('2:',temp_x[np.random.choice(temp_list)].shape)
                        batch_x[1][i] = temp_x[np.random.choice(temp_list)]
                        batch_y[i] = 0 #agregado en esta nueva versión de Gonza/Gabo
                return(batch_x, batch_y)
            
            def nway_one_shot_val(model, n_way, n_val):    
            #    temp_x = x_val
            #    temp_cat_list = cat_test
            #    batch_x=[]
            #    x_0_choice=[]
                n_correct = 0
               
                class_list = np.random.randint(train_size, len(folder_list), n_val)
            
                for i in class_list:  
                    j = np.random.choice(cat_list[i])
                    temp=[]
                    temp.append(np.zeros((n_way, 300, 150,1)))
                    temp.append(np.zeros((n_way, 300, 150,1)))
                    for k in range(0, n_way):
                        temp[0][k] = x[j]            
                        if k==0:
                            # print(i, k, j, np.random.choice(cat_list[i]))
                            temp[1][k] = x[np.random.choice(cat_list[i])]
                        else:
                            # print(i, k, j, np.random.choice(np.append(cat_list[:i].flatten(), cat_list[i+1:].flatten())))
                            temp[1][k] = x[np.random.choice(np.append(cat_list[:i].flatten(), cat_list[i+1:].flatten()))]
            
                    result = model.predict(temp)
                    result = result.flatten().tolist()
                    # print(result)
                    result_index = result.index(min(result))
                    if result_index == 0:
                        n_correct = n_correct + 1
                #print(n_correct, "correctly classified among", n_val)
                # print(temp[0].shape)
                accuracy = (n_correct*100)/n_val
                return accuracy
            
            def nway_one_shot_train(model, n_way, n_val):    
            #    temp_x = x_val
            #    temp_cat_list = cat_test
            #    batch_x=[]
            #    x_0_choice=[]
                n_correct = 0   
                class_list = np.random.randint(0, train_size, n_val)
            
                for i in class_list:  
                    j = np.random.choice(cat_list[i])
                    temp=[]
                    temp.append(np.zeros((n_way, 300, 150,1)))
                    temp.append(np.zeros((n_way, 300, 150,1)))
                    for k in range(0, n_way):
                        temp[0][k] = x[j]            
                        if k==0:
                            #print(i, k, j, np.random.choice(cat_list[i]))
                            temp[1][k] = x[np.random.choice(cat_list[i])]
                        else:
                            #print(i, k, j, np.random.choice(np.append(cat_list[:i].flatten(), cat_list[i+1:].flatten())))
                            temp[1][k] = x[np.random.choice(np.append(cat_list[:i].flatten(), cat_list[i+1:train_size].flatten()))]
            
                    result = model.predict(temp)
                    result = result.flatten().tolist()
                    # print(result)
                    result_index = result.index(min(result))
                    if result_index == 0:
                        n_correct = n_correct + 1
                #print(n_correct, "correctly classified among", n_val)
                # print(temp[0].shape)
                accuracy = (n_correct*100)/n_val
                return accuracy
            
            def contrastive_loss(y, preds, margin=2):
            	# explicitly cast the true class label data type to the predicted
            	# class label data type (otherwise we run the risk of having two
            	# separate data types, causing TensorFlow to error out)
            	y = tf.cast(y, preds.dtype)
            	# calculate the contrastive loss between the true labels and
            	# the predicted labels
            	squaredPreds = K.square(preds)
            	squaredMargin = K.square(K.maximum(margin - preds, 0))
            	loss = K.mean(y * squaredPreds + (1 - y) * squaredMargin)
            	# return the computed contrastive loss to the calling function
            	return loss
            
            batch_size = 128#128
            batch_x, batch_y = get_batch(batch_size)
            # batch_x is a list of 2 elements.
            # Each element contains a number #batch_size sonograms
            # batch_y is 0 is those elements are from the same class
            
            #np.random.randint(train_size, len(folder_list), n_val)
            
            #Building a sequential model
            input_shape=(300, 150, 1)
            left_input = Input(input_shape)
            right_input = Input(input_shape)
            
            W_init = keras.initializers.RandomNormal(mean = 0.0, stddev = 1e-2)
            b_init = keras.initializers.RandomNormal(mean = 0.5, stddev = 1e-2)
            
            #kernel_initializer=W_init
            #bias_initializer=b_init
            
            model = keras.models.Sequential([ 
                keras.layers.Conv2D(32, (5,5),strides=2, activation='relu', input_shape=input_shape, kernel_initializer=W_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
                keras.layers.MaxPooling2D(2, 2),
                keras.layers.Dropout(0.05),
                
                keras.layers.Conv2D(64, (5,5), activation='relu', kernel_initializer=W_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
                keras.layers.MaxPooling2D(2,2),
                keras.layers.Dropout(0.05),
                
                keras.layers.Conv2D(128, (4,4), activation='relu', kernel_initializer=W_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
                keras.layers.MaxPooling2D(1,2),
                keras.layers.Dropout(0.05),
                
                keras.layers.Conv2D(512, (4,4), activation='relu', kernel_initializer=W_init, bias_initializer=b_init, kernel_regularizer=l2(2e-4)),
                keras.layers.MaxPooling2D(2,2),
                keras.layers.Flatten(),
                
                keras.layers.Dense(1024, activation='relu', kernel_initializer=W_init, bias_initializer=b_init,kernel_regularizer=l2(2e-4)),
                # keras.layers.Dense(32, activation='sigmoid', kernel_initializer=W_init, bias_initializer=b_init,kernel_regularizer=l2(1e-3)),
                keras.layers.Dense(salida, activation='linear', kernel_initializer=W_init, bias_initializer=b_init)
            ])#aca se pone la dimensionalidad del espacio de salida
            
            encoded_l = model(left_input)
            encoded_r = model(right_input)
            
            # Add a customized layer to compute the absolute difference between the encodings
            # L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
            L1_layer = Lambda(lambda tensors:tensors[0] - tensors[1])
            
            # subtracted = keras.layers.Subtract()([encoded_l, encoded_r])
            L1_distance = L1_layer([encoded_l, encoded_r])
            
            vector_norm = keras.layers.dot([L1_distance, L1_distance], axes=1)
            # final_layer = Lambda(lambda tensors:K.log(tensors))(vector_norm)
            
            # Add a dense layer with a sigmoid unit to generate the similarity score
            # prediction = Dense(1,activation='sigmoid', use_bias=False)(vector_norm)
            
            siamese_net = Model([left_input, right_input], vector_norm)  # modelo completo
            # distance_net = Model([left_input, right_input], vector_norm)
            encoder = Model([left_input],encoded_l) #codificador para emplear la red
            
            optimizer= Adam(0.0001)
            
            # siamese_net.compile(loss='binary_crossentropy', optimizer=optimizer)
            siamese_net.compile(loss=contrastive_loss, optimizer=optimizer)
            
            # plot_model.plot_model(siamese_net, show_shapes=True, show_layer_names=True)
            # tf.keras.utils.plot_model(model)
            # model.summary()
            
            siamese_net.load_weights(pesos)
            
            ############################################################################################################
            indices_totales = x_train.shape[0]
            lista_coordenadas = []
            lista_clases = []
            for i in range(indices_totales):
                lista_coordenadas.append(encoder.predict([x_train[i].reshape(1,300,150,1)])[0])
                lista_clases.append(y_train[i])
            lista_coordenadas = np.asarray(lista_coordenadas)
            
            indices_totales_val = x_val.shape[0]
            lista_coordenadas_val = []
            lista_clases_val = []
            for i in range(indices_totales_val):
                lista_coordenadas_val.append(encoder.predict([x_val[i].reshape(1,300,150,1)])[0])
                lista_clases_val.append(y_val[i])
            lista_coordenadas_val = np.asarray(lista_coordenadas_val)
            ###########################################################################################################
            #clusteriamos
            X = lista_coordenadas
            cluster_coordenadas = GaussianMixture(n_components=3, random_state=0).fit_predict(X)
            #print('etiquetas de clustering',kmeans_coordenadas.labels_)
            #print('etiquetas de reales',lista_clases)
            X = lista_coordenadas_val
            cluster_coordenadas_val = GaussianMixture(n_components=2, random_state=0).fit_predict(X)
            #print('etiquetas de clustering',kmeans_coordenadas_val.labels_)
            #print('etiquetas de reales',lista_clases_val)
            
            clases_train=cluster_coordenadas == lista_clases
            
            cluster_coordenadas_val= np.where(cluster_coordenadas_val == 0, 3, cluster_coordenadas_val)
            cluster_coordenadas_val= np.where(cluster_coordenadas_val == 1, 4, cluster_coordenadas_val)
            
            clases_val=cluster_coordenadas_val == lista_clases_val
            clases_train_todos=np.concatenate((clases_train_todos, clases_train), axis=0)
            clases_val_todos=np.concatenate((clases_val_todos, clases_val), axis=0)
            
            cantidad_true_train=np.concatenate((cantidad_true_train,[np.sum(clases_train)]), axis=0)
            #cantidad_false_train=np.concatenate((cantidad_false_train, clases_train.count(0)), axis=0)
            cantidad_true_val=np.concatenate((cantidad_true_val,[np.sum(clases_val)]), axis=0)
            #cantidad_false_val=np.concatenate(clases_val.count(0),cantidad_false_val, axis=0)
    
    cantidad_bien_train=np.concatenate((cantidad_bien_train,[np.mean(cantidad_true_train)]), axis=0)
    mediana_cantidad_bien_train=np.concatenate((mediana_cantidad_bien_train,[np.median(cantidad_true_train)]), axis=0)
    cantidad_bien_val=np.concatenate((cantidad_bien_val,[np.mean(cantidad_true_val)]), axis=0)
    mediana_cantidad_bien_val=np.concatenate((mediana_cantidad_bien_val,[np.median(cantidad_true_val)]), axis=0)
    desvio_cantidad_bien_train=np.concatenate((desvio_cantidad_bien_train,[np.std(cantidad_true_train)]), axis=0)
    desvio_cantidad_bien_val=np.concatenate((desvio_cantidad_bien_val,[np.std(cantidad_true_val)]), axis=0)

    print('Para d=',k,':')    
    print('El promedio de train es', np.mean(cantidad_true_train),'+-',np.std(cantidad_true_train))
    print('El promedio de validacion es', np.mean(cantidad_true_val),'+-',np.std(cantidad_true_val))
    print('La mediana de train es', np.median(cantidad_true_train),'+-',np.std(cantidad_true_train))
    print('La mediana de validacion es', np.median(cantidad_true_val),'+-',np.std(cantidad_true_val))



plt.figure(figsize=(10,8))
plt.plot(dimensiones,cantidad_bien_train, '-o', label='puntos train')
#plt.errorbar(x=dimensiones,y=cantidad_bien_val,yerr=desvio_cantidad_bien_val, fmt='x', markersize=5, capsize=6,label='puntos val')
plt.errorbar(x=dimensiones,y=cantidad_bien_train,yerr=desvio_cantidad_bien_train, fmt='o', markersize=5, capsize=6, label='puntos train')
plt.ylabel('Media puntos bien clasif.')
plt.xlabel('Dimensiones')
plt.ylim(20,45)
plt.legend()

#mas linda

plt.figure(figsize=(20,12))
for z in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: 
    
    plt.plot(dimensiones,cantidad_bien_train, '-o',linewidth=1.2)#, label='puntos train')
    plt.plot(dimensiones,cantidad_bien_train +z*desvio_cantidad_bien_train,linewidth=0.8, linestyle='dotted')#, color='slategray')
    plt.plot(dimensiones,cantidad_bien_train -z*desvio_cantidad_bien_train,linewidth=0.8, linestyle='dotted')#, color='slategray')
    plt.plot(dimensiones,cantidad_bien_train +desvio_cantidad_bien_train,linewidth=1, linestyle='dotted')#, color='slategray')
    plt.plot(dimensiones,cantidad_bien_train -desvio_cantidad_bien_train,linewidth=1, linestyle='dotted')#, color='slategray')
    #plt.errorbar(x=dimensiones,y=cantidad_bien_val,yerr=desvio_cantidad_bien_val, fmt='x', markersize=5, capsize=6,label='puntos val')
    #plt.errorbar(x=dimensiones,y=cantidad_bien_train,yerr=desvio_cantidad_bien_train, fmt='o', markersize=5, capsize=6, label='puntos train')
    plt.ylabel('Puntos bien clasif.')
    plt.xlabel('Dimensiones')
    plt.ylim(20,45)
    plt.legend()

#otra version
#sns.lineplot(dimensiones,cantidad_bien_train,hue=desvio_cantidad_bien_train)


#plt.figure(figsize=(10,8))
#dimensiones=[3,4,5,6,7,8,9,10,15]
#plt.errorbar(x=dimensiones,y=mediana_cantidad_bien_train,yerr=desvio_cantidad_bien_train, fmt='o', markersize=5, capsize=6, label='puntos train')
#plt.errorbar(x=dimensiones,y=mediana_cantidad_bien_val,yerr=desvio_cantidad_bien_val, fmt='x', markersize=5, capsize=6, label='puntos val')
#plt.ylabel('Mediana puntos bien clasif.')
#plt.xlabel('Dimensiones')
#plt.legend()

import pandas as pd
#ubicacion_del_archivo='/home/felipe/Documentos/tesis/corridas/vtpgo'
numpy_array=cantidad_bien_train
numpy_array_2=np.around(desvio_cantidad_bien_train,2)
numpy_array_3=cantidad_bien_val
numpy_array_4=desvio_cantidad_bien_val
#df = pd.DataFrame(numpy_array,
 #                index=[dimensiones],
  #               columns=['puntos_bien_train'])


df = {'puntos_bien_train' : numpy_array,
           'dimension' : dimensiones,
           'desvio_train' : numpy_array_2,
           'puntos_bien_val':numpy_array_3,
           'desvio_val':numpy_array_4}

df = pd.DataFrame(df, columns = ['dimension', 'puntos_bien_train', 'desvio_train', 'puntos_bien_val', 'desvio_val'])
df.to_csv(ubicacion_del_archivo,index=False)

#sns.lineplot(data=df, x="dimension", y="puntos_bien_train", hue="desvio_puntos_bien_train")

#%%
dimensiones_nuevas=[3,4,5,6,7,8,9,10,12,15,17,20,30,40,50]
cantidad_bien_train_nuevo=np.delete(cantidad_bien_train, 12)
desvio_cantidad_bien_train_nuevo=np.delete(desvio_cantidad_bien_train, 12)
plt.figure(figsize=(20,12))
for z in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: 
    
    plt.plot(dimensiones_nuevas,cantidad_bien_train_nuevo, '-o',linewidth=1.2)#, label='puntos train')
    plt.plot(dimensiones_nuevas,cantidad_bien_train_nuevo +z*desvio_cantidad_bien_train_nuevo,linewidth=0.8, linestyle='dotted')#, color='slategray')
    plt.plot(dimensiones_nuevas,cantidad_bien_train_nuevo -z*desvio_cantidad_bien_train_nuevo,linewidth=0.8, linestyle='dotted')#, color='slategray')
    plt.plot(dimensiones_nuevas,cantidad_bien_train_nuevo +desvio_cantidad_bien_train_nuevo,linewidth=1, linestyle='dotted')#, color='slategray')
    plt.plot(dimensiones_nuevas,cantidad_bien_train_nuevo -desvio_cantidad_bien_train_nuevo,linewidth=1, linestyle='dotted')#, color='slategray')
    #plt.errorbar(x=dimensiones,y=cantidad_bien_val,yerr=desvio_cantidad_bien_val, fmt='x', markersize=5, capsize=6,label='puntos val')
    #plt.errorbar(x=dimensiones,y=cantidad_bien_train,yerr=desvio_cantidad_bien_train, fmt='o', markersize=5, capsize=6, label='puntos train')
    plt.ylabel('Puntos bien clasif.')
    plt.xlabel('Dimensiones')
    #plt.ylim(20,45)
    plt.legend()