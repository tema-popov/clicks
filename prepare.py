from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import re

matplotlib.style.use('ggplot')

# Read CSV. Convert all date records to ordinal representation

first_cap_re = re.compile('(.)([A-Z][a-z]+)')
all_cap_re = re.compile('([a-z0-9])([A-Z])')
def convert(name):
    s1 = first_cap_re.sub(r'\1_\2', name)
    return all_cap_re.sub(r'\1_\2', s1).lower()

location = r'/home/blumonk/diploma/click.csv'
clicks = read_csv(location).rename(columns=convert)

to_ord = lambda dt: dt.toordinal()
clicks.date = pd.to_datetime(clicks.date).apply(to_ord)
clicks.date_from = pd.to_datetime(clicks.date_from).apply(to_ord)
clicks.date_back = pd.to_datetime(clicks.date_back).apply(to_ord)

# Replace all categorical variables with integral labels

"""
from collections import defaultdict
from sklearn.preprocessing import LabelEncoder

encoders_dict = defaultdict(LabelEncoder)
categorical = ['iata_from', 'iata_to', 'referer', 'flight_class', 
               'iso_from', 'iso_to', 'ak_from', 'ak_to', 'combo_type']
labeled_clicks = clicks.apply(lambda x: encoders_dict[x.name].fit_transform(x.astype(str)) 
                              if x.name in categorical else x)
"""
