import numpy as np
import projectLib as lib
import matplotlib.pyplot as plt

# shape is movie,user,rating
training = lib.getTrainingData()
validation = lib.getValidationData()
ratings_training = training[:,2]
ratinds_validation = training[:,2]
#some useful stats
trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
rBar = np.mean(trStats["ratings"])

# we get the A matrix from the training dataset
def getA(training):
    # A = np.zeros((trStats["n_ratings"], trStats["n_movies"] + trStats["n_users"]))
    # for i in range(0, trStats["n_ratings"] - 1):
    #     A[i][training[i][0]] = training[i][2]
    #     A[i][training[i] + trStats["n_movies"] - 1] = training[i][2]
    # A = np.zeros(trStats["n_users"], trStats["n_movies"])
    # for i in range(0, trStats["n_ratings"]):
    #     A[training[i][1]][training[i][0]] = A[training[i][2]]
    # return A
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
    return (ratings - rBar)

# apply the functions
A = getA(training)
c = getc(rBar, trStats["ratings"])

# compute the estimator b
def param(A, c):
    # ???
    A_t = np.transpose(A)
    inv = np.linalg.inv(np.matmul(A_t, A))
    mul = np.matmul(inv, A_t)
    return np.matmul(mul, c)


# compute the estimator b with a regularisation parameter l
# note: lambda is a Python keyword to define inline functions
#       so avoid using it as a variable name!
def param_reg(A, c, l):
    A_t = np.transpose(A)
    L_i = np.identity(397) * l
    to_inv = np.matmul(A_t, A) + L_i
    inv = np.linalg.inv(to_inv)
    mul = np.matmul(A_t, c)
    return np.matmul(inv, mul)

# from b predict the ratings for the (movies, users) pair
def predict(movies, users, rBar, b):
    n_predict = len(users)
    p = np.zeros(n_predict)
    for i in range(0, n_predict):
        rating = rBar + b[movies[i]] + b[trStats["n_movies"] + users[i]]
        if rating > 5: rating = 5.0
        if rating < 1: rating = 1.0
        p[i] = rating
    return p

#Unregularised version (<=> regularised version with l = 0)
b = param(A, c)

#Regularised version
# l = 0
# b = param_reg(A, c, l)

#print ("Linear regression, l = %f" % l)
print ("RMSE for training %f" % lib.rmse(predict(trStats["movies"], trStats["users"], rBar, b), trStats["ratings"]))
print ("RMSE for validation %f" % lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, b), vlStats["ratings"]))

print(getc(rBar, ratings_training))
RMSE = []
RMSE_val = []
min_RMSE = 1000
min_i = 2500
for i in range(2500,3000,1):
    if (lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, param_reg(A,c,i/1000)), vlStats["ratings"]) < min_RMSE ):
        min_RMSE = lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, param_reg(A,c,i/1000)), vlStats["ratings"])
        min_i = i

    # RMSE.append(lib.rmse(predict(trStats["movies"], trStats["users"], rBar, param_reg(A,c,i/1000)), trStats["ratings"]))
    # RMSE_val.append(lib.rmse(predict(vlStats["movies"], vlStats["users"], rBar, param_reg(A,c,i/1000)), vlStats["ratings"]))

print(min_i)
print (min_RMSE)

#l = 2.531
#min RMSE = 1.0047124491060195

plt.plot(range(20,40,1),RMSE_val)
plt.show()
