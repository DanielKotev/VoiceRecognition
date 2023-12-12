import numpy as np
from xgboost import XGBRegressor
from constants import BASE_DIR
import os
import pathlib
import json
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import mean_absolute_error as MAE


def dataset():
    PARENT_PATH = pathlib.Path(__file__).parent.resolve()
    x_path = os.path.join(PARENT_PATH, BASE_DIR, 'x.csv')
    y_path = os.path.join(PARENT_PATH, BASE_DIR, 'y.csv')

    with open(x_path, 'r') as x_file, open(y_path, 'r') as y_file:
        x, y = [], []
        for _x, _y in zip(x_file, y_file):
            x_line = [float(__x) for __x in _x.strip().split(',')]
            y_line = [float(__y) for __y in _y.strip().split(',')]
            if len(x_line) != 22 or len(y_line) != 22:
                continue
            x.append(x_line)
            y.append(y_line)

    def normalize(matrix, name):
        if os.path.isfile('data.json'):
            with open('data.json', 'r') as f:
                data = json.load(f)
        else:
            data = {}

        mean = np.mean(matrix)
        std_dev = np.std(matrix)

        # Perform Z-normalization
        result = (matrix - mean) / std_dev
        shift = np.abs(np.min(result)) + 1e-14

        # add a key-value pair
        data[f'{name}_mean'] = mean
        data[f'{name}_std_dev'] = std_dev
        data[f'{name}_shift'] = shift

        # save it back to the file
        with open('data.json', 'w') as f:
            json.dump(data, f)

        # for assertion
        with open('data.json', 'r') as f:
            data = json.load(f)

        assert data[f'{name}_mean'] == mean
        assert data[f'{name}_std_dev'] == std_dev
        assert data[f'{name}_shift'] == shift

        # shifting so we won't have negative numbers
        result += shift

        return np.log(result)

    return normalize(np.array(x), 'in'), normalize(np.array(y), 'out')


def train_model():
    X, y = dataset()
    print(X.min(), y.min(), X.max(), y.max())
    # Splitting
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param = {
        'n_estimators': 100,
        'seed': 6833,
        'max_depth': 10,
        'eta': 0.200242512523095,
        'subsample': 0.738395624464073,
        'colsample_bytree': 0.9801272313235927
    }

    if os.path.isfile('model.ubj') or os.path.islink('model.ubj'):
        os.unlink('model.ubj')

    model = XGBRegressor(**param)
    model.fit(X_train, y_train)


    # Predict on test set
    pred = model.predict(X_test)

    mae = MAE(y_test, pred)
    print("MAE : % f" % (mae))

    model.save_model('model.ubj')
    return model


if __name__ == "__main__":
    train_model()
