import json
import numpy as np
from scipy.io import wavfile
import xgboost as xgb
from model_train_csv_all import train_model
from data_prep import fft_data_from_file

fft_noised_first = fft_data_from_file(
    'C:\\Users\\danie\\PycharmProjects\\VoiceRecognition\\dataset\\combine\\v59_n5.wav',
    step=20,
    window=20
)

# If you do not want to train it again you will need to lead the model
# model = train_model()
# create the model
model = xgb.XGBRegressor()

# load the previously saved model
model.load_model('model.ubj')

with open('data.json', 'r') as f:
    data = json.load(f)

# transformation of the input
noised_fft = (fft_noised_first - data['in_mean']) / data['in_std_dev']
noised_fft = np.log(noised_fft + data['in_shift'])

clean_fft = model.predict(noised_fft)
# Remove all normalization and transformation back
clean_fft = (np.exp(clean_fft) - data['out_shift']) * data['out_std_dev'] + data['out_mean']
clean_signal = []
for predicted_window in clean_fft:
    clean_window_fft = [predicted_window[i] + 1j * predicted_window[i + 1] for i in range(0, len(predicted_window), 2)]
    clean_signal.extend(np.fft.irfft(clean_window_fft))

# Save the reconstructed audio to a new .wav file
wavfile.write('C:\\Users\\danie\\PycharmProjects\\VoiceRecognition\\dataset\\output.wav', len(clean_signal),
              np.array(clean_signal).astype(np.int16))
