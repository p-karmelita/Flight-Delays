import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

flights = pd.read_csv('./data/flights.csv', low_memory=False)


flights.drop('WEATHER_DELAY', axis=1, inplace=True)
flights.drop('AIRLINE', axis=1, inplace=True)
flights.drop('FLIGHT_NUMBER', axis=1, inplace=True)
flights.drop('DIVERTED', axis=1, inplace=True)
flights.drop('CANCELLED', axis=1, inplace=True)
flights.drop('CANCELLATION_REASON', axis=1, inplace=True)
flights.drop('YEAR', axis=1, inplace=True)
flights.drop('MONTH', axis=1, inplace=True)
flights.drop('DAY', axis=1, inplace=True)
flights.drop('ORIGIN_AIRPORT', axis=1, inplace=True)
flights.drop('DESTINATION_AIRPORT', axis=1, inplace=True)
flights.drop('WHEELS_ON', axis=1, inplace=True)
flights.drop('WHEELS_OFF', axis=1, inplace=True)
flights.drop('DAY_OF_WEEK', axis=1, inplace=True)
flights.drop('TAIL_NUMBER', axis=1, inplace=True)

null_deptime = flights['DEPARTURE_TIME'].isnull().sum()
null_delay = flights['DEPARTURE_DELAY'].isnull().sum()
null_schedtime = flights['SCHEDULED_TIME'].isnull().sum()
null_elapstime = flights['ELAPSED_TIME'].isnull().sum()
null_airtime = flights['AIR_TIME'].isnull().sum()
null_arrtime = flights['ARRIVAL_TIME'].isnull().sum()
null_arrdelay = flights['ARRIVAL_DELAY'].isnull().sum()
dep_time = null_deptime / len(flights['DEPARTURE_TIME']) * 100
dep_delay = null_delay / len(flights['DEPARTURE_DELAY']) * 100
sched_time = null_schedtime / len(flights['SCHEDULED_TIME']) * 100
elapsed_time = null_elapstime / len(flights['ELAPSED_TIME']) * 100
air_time = null_airtime / len(flights['AIR_TIME']) * 100
arrival_time = null_arrtime / len(flights['ARRIVAL_TIME']) * 100
arrival_delay = null_arrdelay / len(flights['ARRIVAL_DELAY']) * 100

data = [dep_time, dep_delay, sched_time, elapsed_time, air_time, arrival_time, arrival_delay]

df = pd.DataFrame([data], columns=['Czas odlotu', 'Opóźnienie odlotu', 'Zaplanowany czas', 'Czas który upłynął',
                                   'Czas lotu', 'Czas przylotu', 'Opóźnienie przylotu'])

plt.style.use('bmh')

df.plot(kind='bar', figsize=(11, 3), title='Wartość procentowa pustych wartości poszczególnych kolumn.')

arrival_delay = flights['ARRIVAL_DELAY']

max_hours_delay = arrival_delay.max() / 60

min_delay = arrival_delay.min() / 60

data2 = [max_hours_delay, min_delay]
df2 = pd.DataFrame([data2], columns=['Maksymalne opóźnienie', 'Minimalne opóźnienie'])
df2.plot(kind='bar', figsize=(6, 3), title='Maksymalne i minimalne opóźnienie w godzinach.')

flights.dropna(inplace=True)

sched_dep = flights['SCHEDULED_DEPARTURE']
dep_time = flights['DEPARTURE_TIME']
dep_delay = flights['DEPARTURE_DELAY']
sched_time = flights['SCHEDULED_TIME']
elaps_time = flights['ELAPSED_TIME']
air_time = flights['AIR_TIME']
distance = flights['DISTANCE']
sched_arr = flights['SCHEDULED_ARRIVAL']
arr_time = flights['ARRIVAL_TIME']
arr_delay = flights['ARRIVAL_DELAY']


def variance(*values):
    mean = sum(values) / len(values)
    _variance = sum((v - mean) ** 2 for v in values) / len(values)
    return _variance


variance(*arr_delay)
sqrt(1542.2345055354997)

# Wartość wariancji dla poszczególnych kolumn
sched_dep_v = round(233796.44066774342, 4)
dep_time_v = round(246432.59147815884, 4)
dep_delay_v = round(1360.8514780264186, 4)
sched_time_v = round(5672.1972662605685, 4)
elaps_time_v = round(5507.282198783736, 4)
air_time_v = round(5217.290678876618, 4)
distance_v = round(370469.35273315175, 4)
sched_arr_v = round(256948.70146531807, 4)
arr_time_v = round(276647.565786137, 4)
arr_delay_v = round(1542.2345055354997, 4)

# Wartość odchylenia standardowego dla poszczególnych kolumn
sched_dep_sd = round(483.52501555529, 4)
dep_time_sd = round(496.41977345605284, 4)
dep_delay_sd = round(36.88972049265782, 4)
sched_time_sd = round(75.3139911720297, 4)
elaps_time_sd = round(74.21106520448103, 4)
air_time_sd = round(72.23081529981935, 4)
distance_sd = round(608.6619363268511, 4)
sched_arr_sd = round(506.9010766069826, 4)
arr_time_sd = round(525.9729705851214, 4)
arr_delay_sd = round(39.27129365752419, 4)

variance = [sched_dep_v, dep_time_v, dep_delay_v, sched_time_v, elaps_time_v, air_time_v, distance_v,
            sched_arr_v, arr_time_v, arr_delay_v]
stand_dev = [sched_dep_sd, dep_time_sd, dep_delay_sd, sched_time_sd, elaps_time_sd, air_time_sd, distance_sd,
             sched_arr_sd, arr_time_sd, arr_delay_sd]

df = pd.DataFrame([stand_dev], columns=['Planowany odlot', 'Czas odlotu', 'Opóźnienie odlotu', 'Planowany czas',
                                        'Czas trwania lotu', 'Czas w powietrzu', 'Dystans', 'Planowany przylot',
                                        'Czas przylotu', 'Opóźnienie przylotu'])

df.plot(kind='bar', figsize=(9, 4), title='Odchylenie standardowe.')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_absolute_error, # MAE
    mean_squared_error # MSE
)

flights = pd.read_csv('./data/flights.csv')

flights.drop('WEATHER_DELAY', axis=1, inplace=True)
flights.drop('AIRLINE', axis=1, inplace=True)
flights.drop('FLIGHT_NUMBER', axis=1, inplace=True)
flights.drop('DIVERTED', axis=1, inplace=True)
flights.drop('CANCELLED', axis=1, inplace=True)
flights.drop('CANCELLATION_REASON', axis=1, inplace=True)

flights.drop('YEAR', axis=1, inplace=True)
flights.drop('MONTH', axis=1, inplace=True)
flights.drop('DAY', axis=1, inplace=True)
flights.drop('DAY_OF_WEEK', axis=1, inplace=True)
flights.drop('ORIGIN_AIRPORT', axis=1, inplace=True)
flights.drop('DESTINATION_AIRPORT', axis=1, inplace=True)
flights.drop('WHEELS_ON', axis=1, inplace=True)
flights.drop('WHEELS_OFF', axis=1, inplace=True)

flights.drop('TAXI_IN', axis=1, inplace=True)
flights.drop('TAXI_OUT', axis=1, inplace=True)
flights.drop('AIR_SYSTEM_DELAY', axis=1, inplace=True)
flights.drop('SECURITY_DELAY', axis=1, inplace=True)
flights.drop('AIRLINE_DELAY', axis=1, inplace=True)
flights.drop('LATE_AIRCRAFT_DELAY', axis=1, inplace=True)

flights.head()

flights.describe()

flights.isnull().sum()

flights['DEPARTURE_TIME'].value_counts()

flights['DEPARTURE_DELAY'].value_counts()

flights['SCHEDULED_TIME'].value_counts()

null_deptime = flights['DEPARTURE_TIME'].isnull().sum()
len(flights['DEPARTURE_TIME'])
null_delay = flights['DEPARTURE_DELAY'].isnull().sum()
null_schedtime = flights['SCHEDULED_TIME'].isnull().sum()
null_elapstime = flights['ELAPSED_TIME'].isnull().sum()
null_airtime = flights['AIR_TIME'].isnull().sum()
null_arrtime = flights['ARRIVAL_TIME'].isnull().sum()
null_arrdelay = flights['ARRIVAL_DELAY'].isnull().sum()
dep_time = null_deptime / len(flights['DEPARTURE_TIME']) * 100
dep_delay = null_delay / len(flights['DEPARTURE_DELAY']) * 100
sched_time = null_schedtime / len(flights['SCHEDULED_TIME']) * 100
elapsed_time = null_elapstime / len(flights['ELAPSED_TIME']) * 100
air_time = null_airtime / len(flights['AIR_TIME']) * 100
arrival_time = null_arrtime / len(flights['ARRIVAL_TIME']) * 100
arrival_delay = null_arrdelay / len(flights['ARRIVAL_DELAY']) * 100
full_len = len(flights)
full_len
data = [dep_time, dep_delay, sched_time, elapsed_time, air_time, arrival_time, arrival_delay]
df = pd.DataFrame([data], columns=['Czas odlotu','Opóźnienie odlotu', 'Zaplanowany czas', 'Czas który upłynął',
                                 'Czas lotu', 'Czas przylotu', 'Opóźnienie przylotu'])
plt.style.use('bmh')
df.plot(kind='bar', figsize=(11, 3), title='Wartość procentowa pustych wartości poszczególnych kolumn.')

arrival_delay = flights['ARRIVAL_DELAY']
max_hours_delay = arrival_delay.max() / 60
max_hours_delay
min_delay = arrival_delay.min() / 60
min_delay
data2 = [max_hours_delay, min_delay]
df2 = pd.DataFrame([data2], columns=['Maksymalne opóźnienie', 'Minimalne opóźnienie'])
df2.plot(kind='bar', figsize=(6, 3), title='Maksymalne i minimalne opóźnienie w godzinach.')

flights.dropna(inplace=True)

flights.head()

flights.drop('TAIL_NUMBER',axis=1, inplace=True)
flights.head()

flights.describe()

import seaborn as sns

sched_dep = flights['SCHEDULED_DEPARTURE']
dep_time = flights['DEPARTURE_TIME']
dep_delay = flights['DEPARTURE_DELAY']
sched_time = flights['SCHEDULED_TIME']
elaps_time = flights['ELAPSED_TIME']
air_time = flights['AIR_TIME']
distance = flights['DISTANCE']
sched_arr = flights['SCHEDULED_ARRIVAL']
arr_time = flights['ARRIVAL_TIME']
arr_delay = flights['ARRIVAL_DELAY']

def variance(*values):
  mean = sum(values) / len(values)
  _variance = sum((v - mean) ** 2 for v in values) / len(values)
  return _variance

variance(*arr_delay)

# Wartość wariancji dla poszczególnych kolumn
sched_dep_v = round(233796.44066774342, 4)
dep_time_v = round(246432.59147815884, 4)
dep_delay_v = round(1360.8514780264186, 4)
sched_time_v = round(5672.1972662605685, 4)
elaps_time_v = round(5507.282198783736, 4)
air_time_v = round(5217.290678876618, 4)
distance_v = round(370469.35273315175, 4)
sched_arr_v = round(256948.70146531807, 4)
arr_time_v = round(276647.565786137, 4)
arr_delay_v = round(1542.2345055354997, 4)

# Wartość odchylenia standardowego dla poszczególnych kolumn
sched_dep_sd = round(483.52501555529, 4)
dep_time_sd = round(496.41977345605284, 4)
dep_delay_sd = round(36.88972049265782, 4)
sched_time_sd = round(75.3139911720297, 4)
elaps_time_sd = round(74.21106520448103, 4)
air_time_sd = round(72.23081529981935, 4)
distance_sd = round(608.6619363268511, 4)
sched_arr_sd = round(506.9010766069826, 4)
arr_time_sd = round(525.9729705851214, 4)
arr_delay_sd = round(39.27129365752419, 4)

variance = [sched_dep_v, dep_time_v, dep_delay_v, sched_time_v, elaps_time_v, air_time_v, distance_v,
            sched_arr_v, arr_time_v, arr_delay_v]
stand_dev = [sched_dep_sd, dep_time_sd, dep_delay_sd, sched_time_sd, elaps_time_sd, air_time_sd, distance_sd,
            sched_arr_sd, arr_time_sd, arr_delay_sd]

df = pd.DataFrame([stand_dev], columns=['Planowany odlot', 'Czas odlotu', 'Opóźnienie odlotu', 'Planowany czas', 'Czas trwania lotu',
     'Czas w powietrzu', 'Dystans', 'Planowany przylot', 'Czas przylotu', 'Opóźnienie przylotu'])

from scipy import stats

# Współczynnik korelacji Pearsona pomiędzy Czasem odlotu a opóźnieniem odlotu.
departure_correlation = stats.pearsonr(dep_time, dep_delay)

# Współczynnik korelacji Pearsona pomiędzy CZasem przylotu a opóźnieniem przylotu.
arrival_correlation = stats.pearsonr(arr_time, arr_delay)

# Współczynnik korelacji Pearsona pomiędzy Czasem lotu a dystansem.
air_time_dist = stats.pearsonr(air_time, distance)

# from ydata_profiling import ProfileReport
# profile = ProfileReport(flights, title='Profiling report.')
# profile.to_widgets()
# Rozkład zmiennej celu
res = stats.normaltest(arr_delay)

# Test Kołogomorowa-Smirnowa
stats.kstest(arr_delay, stats.norm.cdf)

flights['ARRIVAL_DELAY'].describe()

sns.heatmap(round(flights.corr(), 2), annot=True, linewidths=0.5)

df = pd.DataFrame(flights, columns=['ARRIVAL_DELAY', 'DEPARTURE_DELAY']).sample(n=100000, random_state=1, replace=True)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X = df.values[:, :-1]
Y = df.values[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

model = LinearRegression().fit(X_train, Y_train)
model.intercept_
model.coef_

y = model.score(X_train, Y_train)
z = model.score(X_test, Y_test)
y, z

y_pred = model.predict(X_test)

plt.scatter(X_test, Y_test, color='b', label='Actual Data')
plt.plot(X_test, y_pred, color='r', label='Regression Line')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.title('Linear Regression')
plt.show()

df = pd.DataFrame(flights, columns=['DEPARTURE_TIME', 'DEPARTURE_DELAY', 'SCHEDULED_TIME', 'ARRIVAL_TIME', 'ARRIVAL_DELAY'])

X = df.values[:, :-1]
Y = df.values[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)

model = LinearRegression().fit(X_train, Y_train)
model.intercept_ # wyraz wolny
model.coef_ # współczynnik nachylenia (slope)

y = model.score(X_train, Y_train)
z = model.score(X_test, Y_test)
y, z

y_prediction = model.predict(X_train)
print('MSE on train data = ' , metrics.mean_squared_error(Y_train, y_prediction))

y_pred = model.predict(X_test)