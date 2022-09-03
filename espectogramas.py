import IPython
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
import librosa as lib
from librosa import display
import seaborn as sns
sns.set_theme(style="whitegrid")



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

      S=lib.feature.melspectrogram(y=samples, sr=sampling_rate,n_mels=128,fmax=8000)
      fig, ax = plt.subplots(figsize=(10,5))
      S_dB = lib.power_to_db(S, ref=np.max)
      img = lib.display.specshow(S_dB, x_axis='time',
                              y_axis='mel', sr=sr,
                              fmax=8000, ax=ax)
      plt.axis('off')
      plt.savefig(str(k) + '_' + str(j+1) + '_' + str(i+1) +'.png')
      plt.ioff()
