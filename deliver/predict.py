import json
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
            clicks[new_feature(x)] = (clicks[feature] == x)
        if len(clicks[feature].unique()) > len(top):
            clicks[new_feature('other')] = 1 - sum(clicks[new_feature(col)] for col in top)

    to_ord = lambda dt: dt.toordinal()
    for date_col in [c for c in clicks.columns if c.startswith('date')]:
        clicks[date_col] = pd.to_datetime(clicks[date_col]).apply(to_ord)

    clicks.drop(to_binarize, axis=1, inplace=True)

    with open('scale.json') as f:
        mean_stds = json.load(f)

    for feature in to_scale:
        mu, std = mean_stds[feature]
        clicks[feature] = (clicks[feature] - mu) / std

    print(list(clicks.columns))
    print(clicks[:10])


if __name__ == '__main__':
    prepare_data('click.csv')
