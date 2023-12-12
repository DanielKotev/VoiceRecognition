import os
import random
from os import listdir
from typing import List
from numpy import ndarray
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np


def fft_data_from_file(filename, window=20, step=10):
    try:
        _, data = wavfile.read(filename)
    except:
        return None

    fft_data = []
    i = 0
    while i + step < len(data) - 1:
        x = np.fft.rfft(data[i: i + window])
        _x = np.zeros(x.shape[0] * 2)
        for j in range(x.shape[0]):
            _x[j * 2] = x[j].real + 1e-10
            _x[j * 2 + 1] = x[j].imag + 1e-10
        fft_data.append(_x)
        i += step
    x = np.array(fft_data)
    return x


def convert_ndarray_to_list(fft_data: ndarray):
    list_of_lists = []
    for row in fft_data:
        list_of_lists.append(row.tolist())
    return list_of_lists


def append_lines_to_file(file_path, lines: List[ndarray]):
    with open(file_path, 'a') as file:
        for _line in lines:
            _line = convert_ndarray_to_list(_line)
            file.write(','.join([str(item) for item in _line]) + '\n')


def rename_waves(*paths: str) -> None:
    for path in paths:
        for i, file_name in enumerate(listdir(path=path)):
            source = os.path.join(path, file_name)
            prefix = os.path.basename(path)[0]
            destination = os.path.join(path, f'{prefix}{i}.wav')
            os.rename(source, destination)


def combine_waves(voice_dir, noise_dir, combine_dir):
    for file in listdir(path=voice_dir):
        for i in range(10):
            try:
                sound1 = AudioSegment.from_wav(os.path.join(voice_dir, file))
                noise_wav = random.choice(listdir(noise_dir))
                sound2 = AudioSegment.from_wav(os.path.join(noise_dir, noise_wav))
                sound2 = sound2 - 15
                newone = sound1.overlay(sound2, loop=True)
                newone.export(os.path.join(combine_dir, os.path.splitext(file)[0] + "_" + noise_wav), format='wav')
            except Exception as e:
                print(e)


def _take_max_sec(frame_rate, audio_bytes):
    max_value = audio_bytes.max()
    index = audio_bytes.tolist().index(max_value)
    start_point = int(index / 2)
    if start_point >= 0 and start_point + frame_rate < audio_bytes.shape[0]:
        wav_array = np.array(audio_bytes[start_point:start_point + frame_rate])
    elif index + frame_rate <= audio_bytes.shape[0]:
        wav_array = np.array(audio_bytes[index: index + frame_rate])
    else:
        return None
    return wav_array


def clear_white_noise(old_path, new_path):
    middle_point = 128
    threshold = 1
    frame_rate, audio_bytes = wavfile.read(old_path)
    new_audio = []
    for byte in audio_bytes:
        if byte > threshold + middle_point or byte < middle_point - threshold:
            new_audio.append(byte)
    wav_array = np.array(new_audio)
    wav_array = _take_max_sec(frame_rate, wav_array)
    if wav_array is None or not wav_array.any():
        return
    wavfile.write(new_path, frame_rate, wav_array)