import pandas as pd
import numpy as np

import seaborn as sns
sns.set_style("ticks")

#netflixmovie_df1 = pd.read_csv('D:\\Ozyegin\\2018-Guz\\CS556_BigDataAnalysis\\combined_data_1.txt', header = None, names = ['User_Id', 'User_Rating','TimeStamp'], usecols = [0,1,2])
netflixmovie_df1 = pd.read_csv('D:\\Ozyegin\\2018-Guz\\CS556_BigDataAnalysis\\test.txt', header = None, names = ['User_Id', 'User_Rating','TimeStamp'], usecols = [0,1,2])


#ılk uer 2. movıe
cumulative_df1= pd.DataFrame()
i=0
control=""
user=""
for i in range(len(netflixmovie_df1)):
    #print(netflixmovie_df1.values[i][0] + float(netflixmovie_df1.values[i][1]) + netflixmovie_df1.values[i][2])
    control=str(netflixmovie_df1.values[i][0])
    if control.find(":") > 0:
        user=str(netflixmovie_df1.values[i][0])
        user=user[:-1]
    else:
        s1 = pd.Series([user,netflixmovie_df1.values[i][0],netflixmovie_df1.values[i][1],netflixmovie_df1.values[i][2]])
        tempdf = pd.DataFrame([list(s1)], columns=['User_Id', 'Movie_Id', 'Ratings','Timestamp'])
        cumulative_df1=cumulative_df1.append(tempdf)

print(cumulative_df1)
cumulative_df1['Ratings'] = cumulative_df1['Ratings'].astype(float)
cumulative_df1.index = np.arange(0,len(cumulative_df1))

#Eliminating null Ratings from data
df_nan = pd.DataFrame(pd.isnull(cumulative_df1.Ratings))
df_nan = df_nan[df_nan['Ratings'] == True]
df_nan = df_nan.reset_index()

# remove those Movie ID rows
cumulative_df1 = cumulative_df1[pd.notnull(cumulative_df1['Ratings'])]
cumulative_df1['Movie_Id'] = cumulative_df1['Movie_Id'].astype(int)
cumulative_df1['User_Id'] = cumulative_df1['User_Id'].astype(int)
print(cumulative_df1)
print("")

f = ['count','mean']

df_movie_summary = cumulative_df1.groupby('Movie_Id')['Ratings'].agg(f)
df_movie_summary.index = df_movie_summary.index.map(int)
movie_benchmark = round(df_movie_summary['count'].quantile(0.8),0)
drop_movie_list = df_movie_summary[df_movie_summary['count'] < movie_benchmark].index

df_cust_summary = cumulative_df1.groupby('User_Id')['Ratings'].agg(f)
df_cust_summary.index = df_cust_summary.index.map(int)
cust_benchmark = round(df_cust_summary['count'].quantile(0.8),0)
drop_cust_list = df_cust_summary[df_cust_summary['count'] < cust_benchmark].index

cumulative_df1 = cumulative_df1[~cumulative_df1['Movie_Id'].isin(drop_movie_list)]
cumulative_df1 = cumulative_df1[~cumulative_df1['User_Id'].isin(drop_cust_list)]
print (cumulative_df1)

