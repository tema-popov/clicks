import re
import sys
import time
import json
import _pickle as pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectPercentile
from sklearn.preprocessing import scale

def underscore(name):
    """Convert CamelCase string to python-style underscore_case."""
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def add_dummies(dataset, features, top_ns):
    """Add top-n binary features for each categorical feature.
    
    Arguments:
    dataset -- the DataFrame object
    features -- list of the categorical features to be converted
    top_ns -- list of integers corresponding to the numbers of new binary features
    """
    assert len(features) == len(top_ns)
    copy = dataset.copy()
    tops = {}
    for feature, top_n in zip(features, top_ns):
        grouped = dataset.groupby(feature).size()
        top = grouped.sort_values(ascending=False)[:top_n].index.values.tolist()
        tops[feature] = top
        new_feature = lambda x: '{}_{}'.format(feature, x)
        for x in top:
            copy[new_feature(x)] = (dataset[feature] == x)
        if len(dataset[feature].unique()) > top_n:
            copy[new_feature('other')] = 1 - sum(copy[new_feature(col)] for col in top)
    with open('dummies.json', 'w') as f:
        json.dump(tops, f)
    return copy

to_drop = ['id', 'ak_from', 'ak_to']
to_binarize = ['iata_from', 'iata_to', 'iso_from', 'iso_to', #'ak_from',
               'flight_class', 'combo', 'mobile', 'referer', 'combo_type']
to_scale = ['date', 'price', 'total_price', 'a_days', 'days', 'date_from', 'date_back']

def prepare_data(path):
    """Preprocess dataset:
    1) rename columns
    2) convert dates to ordinals
    3) binarize big categorical features
    4) drop redundant features.
    5) binarize remaining categorical features
    6) scale numerical features
    """
    clicks = pd.read_csv(path).rename(columns=underscore).drop(to_drop, axis=1)

    to_ord = lambda dt: dt.toordinal()
    for date_col in [c for c in clicks.columns if c.startswith('date')]:
        clicks[date_col] = pd.to_datetime(clicks[date_col]).apply(to_ord)

    prepared = add_dummies(clicks, to_binarize, [20] * len(to_binarize))
    prepared.drop(to_binarize, axis=1, inplace=True)
    prepared = prepared[prepared.total_price < 2000000]

    mean_stds = {}
    for feature in to_scale:
        column = prepared[feature].astype(float)
        mean_stds[feature] = (np.mean(column), np.std(column))
        prepared[feature] = scale(column)
    with open('scale.json', 'w') as f:
        json.dump(mean_stds, f)

    return prepared
    
def train(data):
    gbc = GradientBoostingClassifier(random_state=10, n_estimators=20, verbose=1)
    pipe = Pipeline([('selection', SelectPercentile()), ('gbc', gbc)])
    params = {'selection__percentile': range(60, 101, 10), 'gbc__max_depth': range(2, 8, 2)}
    gsearch = GridSearchCV(estimator=pipe, param_grid=params, scoring='neg_log_loss', n_jobs=4, verbose=1, cv=2)
    gsearch.fit(data.drop('book', axis=1), data.book)
    return gsearch.best_estimator_

if __name__ == '__main__':
    start = time.time()
    clicks = prepare_data(sys.argv[1])
    model = train(clicks)
    model.fit(clicks.drop('book', axis=1), clicks.book)
    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)
