from pyboostcard.constraints import *

# m1 = Missing()
# m2 = Exception(-1, 2)
# m3 = Exception(-2, 2)
# m4 = Interval((0.0, 10.0), (True, True), order=3)
# m5 = Interval((10.0, 21.0), (False, True), order=1)

x = np.array([np.nan, -2, -1, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 62, 80, 100])

# c1 = Constraint(m1, m2, m3, m4, m5)


m1 = Missing(order=0)
m2 = Exception(-1, order=2)
m3 = Interval((-2, 18), (True, True), 0, mono=1)
m4 = Interval((18, 62), (False, True), 1, mono=1)
m5 = Interval((62, 100), (False, True), 2, mono=1)

c1 = Constraint(m1, m2, m3, m4, m5)
tf, m = c1.transform(x)


## need to test that this is working

import numpy as np
age = np.random.randint(5, 100, 1000)
probs = np.where(age >= 62, 0.30, np.where(age >= 18, 0.2, 0.1))
y = np.random.binomial(n=1, p=probs)

from xgboost import XGBClassifier
tf, m = c1.transform(age)

clf = XGBClassifier(max_depth=1, n_estimators=100, monotone_constraints=tuple(m), learning_rate=0.1, min_child_weight=50)
clf.fit(tf, y=y, verbose=True)

py = clf.predict_proba(data=tf)[:,1]

import pandas as pd
plt = pd.DataFrame({'x':pd.Series(age), 'y':py})

import seaborn as sns

sns.regplot(x='x', y='y', data=plt, logistic=True)
# plt.plot()

## function to get all of the features and splits of a tree as well as the final points

clf.get_booster().dump_model('xgb_model.txt', with_stats=True)

# read the contents of the file
with open('xgb_model.txt', 'r') as f:
    txt_model = f.read()

print(txt_model)

## regex to get feature #, threshold and leaf values

import re
# trying to extract all patterns like "[f2<2.45]"
splits = re.findall('\[f([0-9]+)<([0-9]+.[0-9-e]+)\]', txt_model)

values = re.findall('leaf=(-{,1}[0-9]+.[0-9-e]+),', txt_model)

## need to implement inverse transformation!? Hope not...

# splits


