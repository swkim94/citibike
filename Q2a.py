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
    """
    :param lat1: start station latitude
    :param long1: start station longitude
    :param lat2: end station latitude
    :param long2: end station longitude
    :return: returns distance in kilometers
    """

    # Reference: http://www.johndcook.com/blog/python_longitude_latitude/
    # Convert latitude and longitude to spherical coordinates in radians.
    degrees_to_radians = math.pi / 180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1) * degrees_to_radians
    phi2 = (90.0 - lat2) * degrees_to_radians

    # theta = longitude
    theta1 = long1 * degrees_to_radians
    theta2 = long2 * degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    # sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

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
# df.loc[m] access each month's dataframe, where m = 1,2,...12
# df.loc[m, i] access ith rows from month m
# df.loc[m].gender or df.loc[m]['gender'] access gender column of month m
# df.iloc[i,j] or df.iloc[i][j] select ith row jth column
# df.iloc[:,k] select kth column, returns pandas series
# df[k] select kth column, returns pandas series
# df[[k]] select kth column, returns pandas dataframe
# df.iloc[i] select ith row, returns pandas series
# df.iloc[[i]] select ith row, returns pandas dataframe

print('=========================  start explore the dataframe and learn about basic facts =========================')
print('exploring the dataframe...')
print('keys to access the columns of the dataframe')
print(df.columns)
# or equivalently,
# df.keys()
# Index(['tripduration', 'starttime', 'stoptime', 'start station id',
#        'start station name', 'start station latitude',
#        'start station longitude', 'end station id', 'end station name',
#        'end station latitude', 'end station longitude', 'bikeid', 'usertype',
#        'birth year', 'gender'],
#       dtype='object')

# df[[5,6,9,10]].hist()   # histogram for the positions of all stations. no surprise here.
# df[['bikeid', 'start station id','end station id', 'tripduration']].hist() # some very long trips. ignore those for the moment
# df_temp = df.head(10000)    # to visualize locations of stations, plot first 100 locations
# P=df_temp.plot(kind='scatter', x='start station longitude', y='start station latitude',s=2,alpha=.6)
# plt.savefig('SK_Map.png')

print('number of unique start station id = %i' % df['start station id'].unique().size)
# number of unique start station id = 488
print('number of unique end station id = %i' % df['end station id'].unique().size)
# number of unique end station id = 498
print('number of bikes in operation for the year 2015 = %i' % df['bikeid'].unique().size)
# number of bikes in operation for the year 2015 = 8477
print('number of bikes in operation during January 2015 = %i' % df.loc[1]['bikeid'].unique().size)
# number of bikes in operation during January 2015 = 4271
n_rides, n_columns =  df.shape
print('total number of rides = %i, number of columns = %i' % (n_rides, n_columns))
# total number of rides = 9937969, number of columns = 15
# df.isnull().sum() reveals that only one column "birth year" has null cells. other cells at least have something.

print('=========================  Q2_1 =========================')
print('Answer to Q2_1: median trip duration = %3.6f' % df['tripduration'].median())
# Answer to Q2_1: median trip duration = 629.000000

print('=========================  Q2_2 =========================')
array = df['start station id'] == df['end station id']
same_start_end = array.sum()
fraction = same_start_end/n_rides
print('Answer to Q2_2: fraction of rides start and end at the same station = %.11f' % fraction)
# Answer to Q2_2: fraction of rides start and end at the same station = 0.0223583913

print('=========================  Q2_3 =========================')
# print(df[['bikeid','start station id']].groupby(['bikeid']).count().head()
array = df[['bikeid','start station id']].groupby(['bikeid']).count()['start station id']
array = array*2 # multiply by 2, because one trip have two visits (start station and end station).
# for sanity check,
# df[['bikeid','start station id']].groupby(['bikeid']).count().sum()
# matches with df.shape[0]

print('Answer to Q2_3: standard deviation of the number of stations visited by a bike = %.7f' % array.std())
# Answer to Q2_3: standard deviation of the number of stations visited by a bike = 657.8235632

print('testing running statistics. looks okay.')
print('benchmark average = %f, std = %.7f' % (array.mean(), array.std()))
print('running average = %f, std = %.7f' % rs1(array.values))

print('=========================  Q2_4 =========================')
print('Beginning data cleanup for Q2_4....')
print('number of trips = %i' % df.shape[0])   # original dataframe has number of trips = 9937969
print('removing trips that start and end at the same station...')
df2 = df.loc[df['start station id'] != df['end station id']]   #define a cleaned dataframe
print('number of trips is reduced to %i' % df2.shape[0])    #  number of trips = 9715772

# remove obviously wrong data -- remove trips that took more than 50000 sec ~ 14 hrs.
# The rate is $15-20/hr, and 8 hrs trip would cost north of $120, so trip this long doesn't make sense.
# Reference: https://www.citibikenyc.com/pricing.
# tripduration histogram looks more reasonable after this cleanup.
print('max trip duration = %i seconds, which is unreasonably long' % df2['tripduration'].max())  # max tripduration is 5864661. NOT reasonable.
print('removing those very long trips...')
cutoff = 50000  # cutoff for tripduration in seconds
df2 = df2.loc[df2['tripduration'] < cutoff]     # reduced to 9571981 records
print('number of trips is reduced to %i with tripduration cutoff %i seconds' % (df2.shape[0], cutoff))
# number of trips is reduced to 9709990 with tripduration cutoff 50000 seconds
# number of trips is reduced to 9649340 with tripduration cutoff 5000 seconds

df2[[5,6,9,10]].hist(bins = 100)   # no surprise here
df2[['bikeid', 'start station id','end station id', 'tripduration']].hist(bins=100)

# distance_on_sphere([40.75001986,-73.96905301,40.72229346,-73.99147535])   # 3.6158197692995766 km. looks good.

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
print('Answer to Q2_4: average length of a trip = %.9f km with tripduration cutoff %i seconds' % (df_temp['distance'].mean(),cutoff))  # benchmark test
print('running average length of a trip = %.9f km with tripduration cutoff %i seconds' % (avg,cutoff))
# Answer to Q2_4: average length of a trip = 1.759229970 km with tripduration cutoff 5000 seconds
# Answer to Q2_4: average length of a trip = 1.759229970 km with tripduration cutoff 50000 seconds

df_temp[df_temp['speed']<50].hist(bins=100)

print('=========================  Q2_5 =========================')
print('just using raw dataframe for Q2_5....')

array = np.zeros(12)
for i in range(1,13):
    array[i-1], ___ = rs1(df.loc[i]['tripduration'])
    print('month = %i, average tripduration = %f seconds'% (i, array[i-1]))

diff = array.max() - array.min()
print('Answer to Q2_5: difference between longest and shortest = %.7f seconds' % diff)
# Answer to Q2_5:difference between longest and shortest = 430.5702960 seconds

print('=========================  Q2_6 =========================')
print('The probelm statement sounds ambiguous. I will assume the question is
       'What is the largest ratio of station hourly usage to system houirly usaage')

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
# Answer to Q2_6: largest ratio of station hourly usage fraction = 0.06562453642

print('=========================  Q2_7 =========================')
print('using the cleaned dataframe from Q2_5...')
# df['usertype'].unique() shows two possible values for this column: array(['Subscriber', 'Customer'], dtype=object)

# (df['usertype'] == 'Subscriber').sum()+(df['usertype'] == 'Customer').sum()  returns 9937969
# which matches with df.shape[0]

sum_customer_overage = ((df['usertype'] == 'Subscriber') & (df['tripduration'] > 45 * 60)).sum()
sum_subscriber_overage = ((df['usertype'] == 'Customer') & (df['tripduration'] > 30 * 60)).sum()
sum = sum_customer_overage + sum_subscriber_overage
fraction = sum/df.shape[0]

print('Answer to Q2_7: fraction of rides exceed their corresponding time limit = %.11f' % fraction)
# Answer to Q2_7: fraction of rides exceed their corresponding time limit = 0.03810678017

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
# Answer to Q2_8: average number of times a bike is moved = 133.618379