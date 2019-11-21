from pyspark.mllib.recommendation import ALS
from pyspark import SparkConf, SparkContext
from pyspark.sql import Row
from math import sqrt
from timeit import default_timer as timer

conf = SparkConf().setAppName("ALS Project")
sc = SparkContext(conf=conf)

moviedata = sc.textFile("D:\\Ozyegin\\2018-Guz\\CS556_BigDataAnalysis\\ratings.dat")

#Split data 
splitted_data = moviedata.map(lambda x:x.split('::'))
#splitted_data = moviedata.map(lambda x:x.split(','))

#Calculating the mean o all ratings
rate = splitted_data.map(lambda y: float(y[2]))
rate.mean()
print(rate.mean())


#Extract just the users
users = splitted_data.map(lambda y: int(y[0]))
users_count=users.distinct().count()

#Extract just the movies
movies_count=splitted_data.map(lambda y: int(y[1])).distinct().count()

print("Distinct User Count is : %d"%users_count)
print("Distinct Movies Count is : %d"%movies_count)



#(user, item, rating)
mls = moviedata.map(lambda l: l.split('::'))
ratings = mls.map(lambda x: Row(int(x[0]),int(x[1]), float(x[2])))

#Splitting data into 2 for  training and test set
train, test = ratings.randomSplit([0.8,0.2],7856)

print(train.count())
print(test.count())

#Caching data for speed
train.cache()
test.cache()

#Setting up the parameters for ALS, they are most efficients
rank = 5 # Latent Factors to be made
numIterations = 10# Times to repeat process

print("Creating a model for Training with ALS Movie Recommendation")
print("___________________________________________________________")
model = ALS.train(train, rank, numIterations)

print("")
print("")
print("_______________________________________________")
print("Recommend Movie X to N counted User Like Below:")
print("_______________________________________________")
return_val=model.recommendUsers(286,10)
for val in range(len(return_val)):
    print(return_val[val])

print("")
print("")
print("Recommend Movies to User X :")
print("_______________________________________________")
return_val=model.recommendProducts(222,10)
for val in range(len(return_val)):
    print(return_val[val])


#Predict Single Product for Single User
return_val=model.predict(222, 286)



# Predict Multi Users and Multi Products
# Pre-Processing
pred_input = train.map(lambda x:(x[0],x[1]))

# Lots of Predictions
#Returns Ratings(user, item, prediction)
pred = model.predictAll(pred_input)
start = timer()
#Get Performance Estimate
#Organize the data to make (user, product) the key)
true_reorg = train.map(lambda x:((x[0],x[1]), x[2]))
pred_reorg = pred.map(lambda x:((x[0],x[1]), x[2]))

#Do the actual join
true_pred = true_reorg.join(pred_reorg)
print("")
print("")
print("_______________________________________________")
MSE = true_pred.map(lambda r: (r[1][0] - r[1][1])**2).mean()
RMSE = sqrt(MSE)
print("Train MSE:%2.4f",MSE)
print("Train RMSE:%2.4f",RMSE)

#Test Set Evaluation
#More dense, but nothing we haven't done before
test_input = test.map(lambda x:(x[0],x[1]))
pred_test = model.predictAll(test_input)
test_reorg = test.map(lambda x:((x[0],x[1]), x[2]))
pred_reorg = pred_test.map(lambda x:((x[0],x[1]), x[2]))
test_pred = test_reorg.join(pred_reorg)
test_MSE = test_pred.map(lambda r: (r[1][0] - r[1][1])**2).mean()
test_RMSE = sqrt(test_MSE)
print("_______________________________________________")
print("Test MSE:%2.4f",test_MSE)
print("Test RMSE:%2.4f",test_RMSE)
duration = timer() - start
print("_______________________________________________")
print("Time for Duration:%2.4f",duration)