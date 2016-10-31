import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import math

def rs1(n):
    """
    single pass, running statistics version (Welford algorithm), Reference: http://www.johndcook.com/blog/standard_deviation/
    this function is expecting to receive a very large array and return running average and standard deviation.
    """
    if n.size == 1:
        return n[0], 0
    else:
        k = 0
        M = n[0]
        S = 0.0
        for ni in n:
            k += 1
            M_old = M
            M = M_old + (ni - M_old) / k
            S = S + (ni - M_old) * (ni - M)
        var = S / (k-1)
        return M, var**0.5

def distance_on_sphere(lat1, long1, lat2, long2):
    # Reference: http://www.johndcook.com/blog/python_longitude_latitude/
    degrees_to_radians = math.pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    # theta = longitude
    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    cos = (math.sin(phi1) * math.sin(phi2) * math.cos(theta1 - theta2) +
           math.cos(phi1) * math.cos(phi2))
    arc = math.acos(cos)
    R = 6371  # Radius of the Earth in km
    distance = arc * R

    return distance

print('=========================  begin Import data, total 1.73 GB, csv format =========================')
print('reading data...')
df = pd.read_csv('201501-citibike-tripdata.csv')
n_rides, n_columns =  df.shape
arrays = [[1]*n_rides, range(n_rides)]
df.index = pd.MultiIndex.from_arrays(arrays, names=['month','num'])
print('finished reading 201501-citibike-tripdata.csv')
for i in range(2,13):
    filename = str(201500+i)+'-citibike-tripdata.csv'
    df_temp = pd.read_csv(filename)
    n_rides, n_columns = df_temp.shape
    arrays = [[i] * n_rides, range(n_rides)]
    df_temp.index = pd.MultiIndex.from_arrays(arrays, names=['month', 'num'])
    df = pd.concat([df, df_temp])
    print('finished reading '+filename)

print('=========================  start explore the dataframe and learn about basic facts =========================')
print('exploring the dataframe...')
print('keys to access the columns of the dataframe')
print(df.columns)
print('number of unique start station id = %i' % df['start station id'].unique().size)
print('number of unique end station id = %i' % df['end station id'].unique().size)
print('number of bikes in operation for the year 2015 = %i' % df['bikeid'].unique().size)
print('number of bikes in operation during January 2015 = %i' % df.loc[1]['bikeid'].unique().size)
n_rides, n_columns =  df.shape
print('total number of rides = %i, number of columns = %i' % (n_rides, n_columns))

print('=========================  Q2_1 =========================')
print('Answer to Q2_1: median trip duration = %3.6f' % df['tripduration'].median())

print('=========================  Q2_2 =========================')
array = df['start station id'] == df['end station id']
same_start_end = array.sum()
fraction = same_start_end/n_rides
print('Answer to Q2_2: fraction of rides start and end at the same station = %.11f' % fraction)

print('=========================  Q2_3 =========================')
array = df[['bikeid','start station id']].groupby(['bikeid']).count()['start station id']
array = array*2 # multiply by 2, because one trip have two visits (start station and end station).
print('Answer to Q2_3: standard deviation of the number of stations visited by a bike = %.7f' % array.std())

print('=========================  Q2_4 =========================')
print('Beginning data cleanup for Q2_4....')
print('number of trips = %i' % df.shape[0])   # original dataframe has number of trips = 9937969
print('removing trips that start and end at the same station...')
df2 = df.loc[df['start station id'] != df['end station id']]   #define a cleaned dataframe
print('number of trips is reduced to %i' % df2.shape[0])    #  number of trips = 9715772

print('max trip duration = %i seconds, which is unreasonably long' % df2['tripduration'].max())  # max tripduration is 5864661. NOT reasonable.
print('removing those very long trips...')
cutoff = 50000  # cutoff for tripduration in seconds
df2 = df2.loc[df2['tripduration'] < cutoff]     # reduced to 9571981 records
print('number of trips is reduced to %i with tripduration cutoff %i seconds' % (df2.shape[0], cutoff))

df2[[5,6,9,10]].hist(bins = 100)   # no surprise here
df2[['bikeid', 'start station id','end station id', 'tripduration']].hist(bins=100)

distances = np.zeros(0)
for i in range(1,13):
    df_temp = df2.loc[i][['start station latitude', 'start station longitude','end station latitude', 'end station longitude']]
    print('calculating distances for month %i' % i)
    array = np.zeros(df_temp.shape[0])
    array = df_temp[['start station latitude', 'start station longitude','end station latitude', 'end station longitude']].apply(lambda x: distance_on_sphere(*x), axis=1)
    distances = np.append(distances,array)

speed = distances/df2['tripduration']*60*60
print('max bike speed was found to be %f km/h' % speed.max())
print('removing data with speed >50 km/hr or >31.25 mph, these must be errors or transportation by a car(?)')
print('those accounted for %f per cent of the total trips' % ((speed > 50).sum()/speed.size*100))
df_temp = pd.DataFrame(distances, columns=['distance'])
df_temp = df_temp.assign(speed = speed.values)
df_temp[df_temp['speed']<50]

avg, ___ = rs1(df_temp['distance']) # running average of the distances
print('running average length of a trip = %.9f km with tripduration cutoff %i seconds' % (avg,cutoff))
print('Answer to Q2_4: average length of a trip = %.9f km with tripduration cutoff %i seconds' % (df_temp['distance'].mean(),cutoff))  # benchmark test
df_temp[df_temp['speed']<50].hist(bins=100)

print('=========================  Q2_5 =========================')
print('just using raw dataframe for Q2_5....')

array = np.zeros(12)
for i in range(1,13):
    array[i-1], ___ = rs1(df.loc[i]['tripduration'])
    print('month = %i, average tripduration = %f seconds'% (i, array[i-1]))

diff = array.max() - array.min()
print('Answer to Q2_5: difference between longest and shortest = %.7f seconds' % diff)

print('=========================  Q2_6 =========================')
df_temp = df.loc[1][['starttime','start station id']]
print('processing month 1 data for hourly usage fraction. This takes some time...')
df_temp['starttime'] = pd.to_datetime(df_temp['starttime'])
df_temp.set_index('starttime', inplace=True)
df2 = df_temp.groupby([df_temp.index.hour, df_temp['start station id']]).size()
for i in range(2,13):
    df_temp = df.loc[i][['starttime','start station id']]
    print('processing month %i data for hourly usage fraction. This takes some time...' % i)
    df_temp['starttime'] = pd.to_datetime(df_temp['starttime'])
    df_temp.set_index('starttime', inplace=True)
    df2 = df2.add(df_temp.groupby([df_temp.index.hour, df_temp['start station id']]).size(), fill_value=0)
fraction = np.ones(24)
for i in range(0,24):
    fraction[i] = df2.loc[i].max()/df2.loc[i].sum()
    print('max hourly usage fraction for %i:00-%i:00 =%f' % (i, i+1, fraction[i]))
plt.figure()
plt.plot(fraction, 'ko-')
plt.title('max hourly usage fraction for each hour')
print('Answer to Q2_6: largest ratio of station hourly usage fraction = %.11f' % fraction.max())

print('=========================  Q2_7 =========================')
print('using the cleaned dataframe from Q2_5...')
sum_customer_overage = ((df['usertype'] == 'Subscriber') & (df['tripduration'] > 45 * 60)).sum()
sum_subscriber_overage = ((df['usertype'] == 'Customer') & (df['tripduration'] > 30 * 60)).sum()
sum = sum_customer_overage + sum_subscriber_overage
fraction = sum/df.shape[0]
print('Answer to Q2_7: fraction of rides exceed their corresponding time limit = %.11f' % fraction)

print('=========================  Q2_8 =========================')
df_temp = df.reset_index()[['bikeid','starttime','start station id','end station id']]
df_temp = df_temp.sort_values(['bikeid','starttime'])
df2 = df_temp.groupby('bikeid')
bikeids = np.zeros(0)
num_moved = np.zeros(0)
for name, group in df2:
    bikeids = np.append(bikeids, name)
    num = (group['start station id'] != group['end station id'].shift(1)).sum() -1
    num_moved = np.append(num_moved, num)
print('Answer to Q2_8: average number of times a bike is moved = %f' % num_moved.mean())

print('=========================  end =========================')