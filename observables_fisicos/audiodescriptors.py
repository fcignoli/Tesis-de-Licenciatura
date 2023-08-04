#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 20:42:23 2022

@author: felipe
"""

import IPython
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import librosa as lib
from librosa import display
import seaborn as sns
sns.set_theme(style="whitegrid")

centroides=[]
roll=[]
flatness = []
bandwidth=[]
rms=[]

for k in ['violin','guitarra', 'oboe', 'piano', 'trompeta']:
  for j in np.arange(2):
    for i in np.arange(7):

      nombre = '/home/felipe/Documentos/tesis/datos_entrenamiento/'
      sufijo = '.wav'
      completo = nombre + str(k) + '_' + str(j+1) + '_' + str(i+1) + sufijo
      
      file_path = completo
      
      samples, sampling_rate = lib.load(file_path)
        
      duration_of_sound = len(samples)/sampling_rate
      sr=sampling_rate
      time= np.linspace(0,duration_of_sound,len(samples))
      cent = lib.feature.spectral_centroid(y=samples, sr=sr)
      flat=lib.feature.spectral_flatness(y=samples)
      time_cent = lib.times_like(cent)
      rolloff = lib.feature.spectral_rolloff(y=samples, sr=sr, roll_percent=0.95) #en timbre tool box usan 95%
      time_roll = lib.times_like(rolloff)
      spec_bw = lib.feature.spectral_bandwidth(y=samples, sr=sr)
      time_spec_bw=lib.times_like(spec_bw)
      rms_muestra=librosa.feature.rms(y=samples)
      rms=np.concatenate((rms,np.mean(rms_muestra)), axis=None)
      bandwidth=np.concatenate((bandwidth,np.mean(spec_bw[0])), axis=None)
      centroides=np.concatenate((centroides,np.mean(cent[0])), axis=None)
      roll=np.concatenate((roll,np.mean(rolloff[0])), axis=None)
      flatness = np.concatenate((flatness,np.mean(flat[0])), axis=None)


np.savetxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/Rolloff.txt',roll,fmt = '%10.5f') #los guardo
np.savetxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/Centroides.txt',centroides,fmt = '%10.5f')
np.savetxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/Bandwidth.txt',bandwidth,fmt = '%10.5f')
np.savetxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/Flatness.txt',flatness,fmt = '%10.5f')
np.savetxt('/home/felipe/Documentos/tesis/audiodescriptores_cada_muestra/rms.txt',rms,fmt = '%10.5f')