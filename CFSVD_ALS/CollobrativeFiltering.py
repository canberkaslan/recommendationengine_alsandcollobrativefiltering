import pandas as pd
import numpy as np
from surprise import Reader, Dataset, SVD, evaluate
from surprise.model_selection import cross_validate, train_test_split
from surprise import accuracy
from timeit import default_timer as timer
# Skip date
header = ['User_Id','Movie_Id','User_Rating','Timestamp']
#netflixmovie_df1 = pd.read_csv('C:\\Users\\canberk\\Downloads\\ml-10m\\ml-10M100K\\ratings.dat', header = None, names = ['User_Id','Movie_Id','User_Rating','Timestamp'], usecols = [0,1,2,3])
netflixmovie_df1 = pd.read_table(r'D:\\Ozyegin\\2018-Guz\\CS556_BigDataAnalysis\\ratings.dat', sep='::',header = None, names = header, usecols = [0,1,2,3])

start = timer()
netflixmovie_df1['User_Rating'] = netflixmovie_df1['User_Rating'].astype(float)

#For performance, getting only one txt data
sumof_moviedata = netflixmovie_df1

#Indexing Data
sumof_moviedata.index = np.arange(0,len(sumof_moviedata))


df_title = pd.read_table(r'D:\\Ozyegin\\2018-Guz\\CS556_BigDataAnalysis\\movies.dat', sep='::', header = None, names = ['Movie_Id','Movie_Name','Definition'])
df_title.set_index('Movie_Id', inplace = True)

print(df_title.head(10))

reader = Reader()

# For 100k Data applying SVD
data = Dataset.load_from_df(sumof_moviedata[['User_Id','Movie_Id','User_Rating']][:100000], reader)
algorithm = SVD()
cross_validate(algorithm, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

trainset, testset = train_test_split(data, test_size=.20)

#Train the algorithm on the trainset, and predict ratings for the testset
algorithm.fit(trainset)
predictions = algorithm.test(testset)

# Then compute RMSE
accuracy.rmse(predictions)
predictions = algorithm.fit(trainset).test(testset)
duration = timer() - start
print("Time for Duration:%2.4f",duration)

is_count = 0
while not is_count:
    choosed_c = int(input('Please enter a Customer ID'))
    if(choosed_c >=1 ):
        is_count = 1  ## set it to 1 to validate input and to terminate the while..not loop
    else:
        print("'%s' is not a valid integer. Please enter valid Customer ID"%choosed_c)

df_user = sumof_moviedata[(sumof_moviedata['User_Id'] == choosed_c) & (sumof_moviedata['User_Rating'] == 5)]
df_user = df_user.set_index('Movie_Id')
df_user = df_user.join(df_title)['Movie_Name']
print(df_user)

df_user = df_title.copy()
df_user = df_user.reset_index()
data = Dataset.load_from_df(sumof_moviedata[['User_Id','Movie_Id','User_Rating']], reader)
trainset = data.build_full_trainset()
algorithm.fit(trainset)

df_user['Estimate_Score'] = df_user['Movie_Id'].apply(lambda x: algorithm.predict(choosed_c, x).est)

df_user = df_user.drop('Movie_Id', axis = 1)

df_user = df_user.sort_values('Estimate_Score', ascending=False)
print(df_user.head(30))