import uuid
import pathlib
import os
from constants import BASE_DIR, COMBINED_DIR, NOISE_DIR, VOICE_DIR, RAW_VOICE_DIR
from data_prep import clear_white_noise, rename_waves, combine_waves, fft_data_from_file, append_lines_to_file

if __name__ == "__main__":

    PARENT_PATH = pathlib.Path(__file__).parent.resolve()
    noise_dir = os.path.join(PARENT_PATH, BASE_DIR, NOISE_DIR)
    voice_dir = os.path.join(PARENT_PATH, BASE_DIR, VOICE_DIR)
    raw_voice_dir = os.path.join(PARENT_PATH, BASE_DIR, RAW_VOICE_DIR)
    combined_dir = os.path.join(PARENT_PATH, BASE_DIR, COMBINED_DIR)

    all_dirs = [noise_dir, voice_dir, combined_dir]

    #removing all files in combined_dir
    for _dir in [combined_dir, voice_dir]:
        for filename in os.listdir(_dir):
            file_path = os.path.join(_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

    for root, dirs, files in os.walk(raw_voice_dir):
        print(f'Processing {root}')
        for file in files:
            if not file.endswith(".wav"):
                continue
            new_path = os.path.join(voice_dir, str(uuid.uuid4()) + '.wav')
            clear_white_noise(os.path.join(root, file), new_path)

    rename_waves(voice_dir, noise_dir)

    combine_waves(voice_dir, noise_dir, combined_dir)

    for file_name in os.listdir(combined_dir):
        combined_file_path = os.path.join(combined_dir, file_name)
        X_fft = fft_data_from_file(combined_file_path)
        voice_file_path = os.path.join(voice_dir, file_name.split('_')[0] + '.wav')
        y_fft = fft_data_from_file(voice_file_path)

        assert X_fft.shape[0] == y_fft.shape[0]

        append_lines_to_file(os.path.join(PARENT_PATH, BASE_DIR, 'x.csv'), X_fft)
        append_lines_to_file(os.path.join(PARENT_PATH, BASE_DIR, 'y.csv'), y_fft)