import sys
import json
import _pickle as pickle
import pandas as pd
from train import to_drop, to_binarize, to_scale, underscore

def prepare_data(path):
    """Convert data to match the format used on training.
    """
    clicks = pd.read_csv(path).rename(columns=underscore).drop(to_drop, axis=1)
    with open('dummies.json') as f:
        tops = json.load(f)
    for feature in to_binarize:
        new_feature = lambda x: '{}_{}'.format(feature, x)
        top = tops[feature]
        for x in top:
            if x == 'other':
                clicks[new_feature('other')] = 1 - sum(clicks[new_feature(col)] for col in top[:-1])
            else:
                clicks[new_feature(x)] = (clicks[feature] == x)
    to_ord = lambda dt: dt.toordinal()
    for date_col in [c for c in clicks.columns if c.startswith('date')]:
        clicks[date_col] = pd.to_datetime(clicks[date_col]).apply(to_ord)

    clicks.drop(to_binarize, axis=1, inplace=True)

    with open('scale.json') as f:
        mean_stds = json.load(f)

    for feature in to_scale:
        mu, std = mean_stds[feature]
        clicks[feature] = (clicks[feature] - mu) / std if std != 0 else 0
    return clicks

if __name__ == '__main__':
    clicks = prepare_data(sys.argv[1])
    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)
    probs = model.predict_proba(clicks.drop('book', axis=1))[:,1]
    with open(sys.argv[2], 'w') as f:
        f.write("\n".join([str(prob) for prob in list(probs)]))
