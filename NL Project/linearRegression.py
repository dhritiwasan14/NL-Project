import numpy as np
import projectLib as lib
import matplotlib.pyplot as plt

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()

#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])
ratings = trStats["ratings"]
# we get the A matrix from the training dataset
def getA(training):
    A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    no_movies = trStats["n_movies"]
    data = lib.getTrainingData()
    for i, rating in enumerate(data):
        movie, user, r = rating
        A[i][movie] = 1
        A[i][no_movies+user] = 1
    return A

# we also get c
def getc(rBar, ratings):
    c = ratings - rBar
    return c

# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    A_t = np.transpose(A)
    inv = np.linalg.inv(np.matmul(A_t, A))
    b = np.matmul(np.matmul(inv, np.transpose(A)), c)
    return b

# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    # print(np.matmul(np.transpose(A), A).shape)
    reg = np.multiply(np.identity(397), l)
    A_t = A.transpose()
    inv = np.linalg.inv(np.matmul(A_t, A)+reg)
    b = np.matmul(np.matmul(inv, np.transpose(A)), c)
    return b

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        print(movies[i], users[i])
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p



# Unregularised version (<=> regularised version with l = 0)
A = getA('training.csv')
c = getc(rBar, ratings)
b = param(A, c)
rtrain = lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])
rval = lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"])
print('UNREGULARISED RMSE FOR TRAINING ', rtrain, '; UNREGULARISED RMSE FOR VALIDATION ',rval)

# Regularised version
rmse_training = []
rmse_validation = []
lambda_vals = []
for l in range(2500, 2550):
    l = l/1000.0
    lambda_vals.append(l)
    b = param_reg(A, c, l)
    rtrain = lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"])
    rval = lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"])
    rmse_training.append(rtrain)
    rmse_validation.append(rval)

RMSE_VALIDATION_MIN = min(rmse_validation)
print('MINIMUM RMSE FOR VALIDATION ', RMSE_VALIDATION_MIN)
print('LAMBDA WITH MINIMUM RMSE FOR VALIDATION ', (2500+rmse_validation.index(RMSE_VALIDATION_MIN))/1000.0)


no_users = trStats["n_users"]
no_movies = trStats["n_movies"]
ratings = []

for i in range(no_users):
    temp = []
    for j in range(no_movies):
        rating = rBar + b[j] + b[trStats["n_movies"] + i]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        temp.append(rating)

    ratings.append(temp)

ratings = np.asarray(ratings)
np.savetxt("linearRegression.txt", ratings)
