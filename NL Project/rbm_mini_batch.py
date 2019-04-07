import numpy as np
import projectLib as lib

# set highest rating
K = 5

def softmax(x):
    # Numerically stable softmax function
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def ratingsPerMovie(training):
    movies = [x[0] for x in training]
    u_movies = np.unique(movies).tolist()
    return np.array([[i, movie, len([x for x in training if x[0] == movie])] for i, movie in enumerate(u_movies)])

def getV(ratingsForUser):
    # ratingsForUser is obtained from the ratings for user library
    # you should return a binary matrix ret of size m x K, where m is the number of movies
    #   that the user has seen. ret[i][k] = 1 if the user
    #   has rated movie ratingsForUser[i, 0] with k stars
    #   otherwise it is 0
    ret = np.zeros((len(ratingsForUser), K))
    for i in range(len(ratingsForUser)):
        ret[i, ratingsForUser[i, 1]-1] = 1.0
    return ret

def getInitialWeights(m, F, K):
    # m is the number of visible units
    # F is the number of hidden units
    # K is the highest rating (fixed to 5 here)
    return np.random.normal(0, 0.1, (m, F, K))

def sig(x):
    ### TO IMPLEMENT ###
    # x is a real vector of size n
    # ret should be a vector of size n where ret_i = sigmoid(x_i)
    n = x.shape[0]
    ret = np.zeros(n)
    for i in range(0,n):
        ret[i] = 1.0/(1 + np.exp(-x[i]))
    return ret

def visibleToHiddenVec(v, w):
    ### TO IMPLEMENT ###
    # v is a matrix of size m x 5. Each row is a binary vector representing a rating
    #    OR a probability distribution over the rating
    # w is a list of matrices of size m x F x 5
    # ret should be a vector of size F
    F = w.shape[1]
    m = v.shape[0]
    ret = np.zeros((F))
    for j in range(F):
        sum = 0
        for k in range(5):
            for i in range(m):
                sum += v[i][k]* w[i][j][k]
        ret[j] = sum

    return sig(ret)

def hiddenToVisible(h, w):
    ### TO IMPLEMENT ###
    # h is a binary vector of size F
    # w is an array of size m x F x 5
    # ret should be a matrix of size m x 5, where m
    #   is the number of movies the user has seen.
    #   Remember that we do not reconstruct movies that the user
    #   has not rated! (where reconstructing means getting a distribution
    #   over possible ratings).
    #   We only do so when we predict the rating a user would have given to a movie.
    m = w.shape[0]
    F = h.shape[0]
    ret = np.zeros((m,5))
    for i in range (m):
        for k in range(5):
            sum = 0
            for j in range (F):
                sum = sum + h[j]*w[i][j][k]
            ret[i][k] = sum
        ret[i] = sig(ret[i])
    return ret

def probProduct(v, p):
    # v is a matrix of size m x 5
    # p is a vector of size F, activation of the hidden units
    # returns the gradient for visible input v and hidden activations p
    ret = np.zeros((v.shape[0], p.size, v.shape[1]))
    for i in range(v.shape[0]):
        for j in range(p.size):
            for k in range(v.shape[1]):
                ret[i, j, k] = v[i, k] * p[j]
    return ret

def sample(p):
    # p is a vector of real numbers between 0 and 1
    # ret is a vector of same size as p, where ret_i = Ber(p_i)
    # In other word we sample from a Bernouilli distribution with
    # parameter p_i to obtain ret_i

    samples = np.random.random(p.size)
    return np.array(samples <= p, dtype=int)

def getPredictedDistribution(v, w, wq):
    ### TO IMPLEMENT ###
    # This function returns a distribution over the ratings for movie q, if user data is v
    # v is the dataset of the user we are predicting the movie for
    #   It is a m x 5 matrix, where m is the number of movies in the
    #   dataset of this user.
    # w is the weights array for the current user, of size m x F x 5
    # wq is the weight matrix of size F x 5 for movie q
    #   If W is the whole weights array, then wq = W[q, :, :]
    # You will need to perform the same steps done in the learning/unlearning:
    #   - Propagate the user input to the hidden units
    #   - Sample the state of the hidden units
    #   - Backpropagate these hidden states to obtain
    #       the distribution over the movie whose associated weights are wq
    # ret is a vector of size 5
    p = visibleToHiddenVec(v, w)
    neg_data = hiddenToVisible(sample(p), np.array([wq]))
    return neg_data[0,:]


def predictRatingMax(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the one with the highest probability
    return (np.argmax(ratingDistribution) + 1)

def predictRatingMean(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the expectation over ratingDistribution
    sum = np.sum(ratingDistribution)
    norm_prob = ratingDistribution/sum
    ret = 0
    for i in range (len(ratingDistribution)):
        ret += (i+1)*norm_prob[i]
    return ret

def predictRatingExp(ratingDistribution):
    ### TO IMPLEMENT ###
    # ratingDistribution is a probability distribution over possible ratings
    #   It is obtained from the getPredictedDistribution function
    # This function is one of three you are to implement
    # that returns a rating from the distribution
    # We decide here that the predicted rating will be the expectation over
    # the softmax applied to ratingDistribution
    sum_exp = np.sum(np.exp(ratingDistribution))
    prob = np.exp(ratingDistribution)/sum_exp
    ret = 0
    for i in range (len(ratingDistribution)):
        ret += (i+1)*prob[i]
    return ret

def predictMovieForUser(q, user, W, training, predictType="mean"):
    # movie is movie idx
    # user is user ID
    # type can be "max" or "exp"
    ratingsForUser = lib.getRatingsForUser(user, training)
    v = getV(ratingsForUser)
    ratingDistribution = getPredictedDistribution(v, W[ratingsForUser[:, 0], :, :], W[q, :, :])
    if predictType == "max":
        return predictRatingMax(ratingDistribution)
    elif predictType == "mean":
        return predictRatingMean(ratingDistribution)
    else:
        return predictRatingExp(ratingDistribution)

def predict(movies, users, W, training, predictType="mean"):
    # given a list of movies and users, predict the rating for each (movie, user) pair
    # used to compute RMSE
    return [predictMovieForUser(movie, user, W, training, predictType=predictType) for (movie, user) in zip(movies, users)]

def predictForUser(user, W, training, predictType="mean"):
    ### TO IMPLEMENT
    # given a user ID, predicts all movie ratings for the user
    predictions = np.zeros((W.shape[0]))
    for i in range(W.shape[0]):
        predictions[i] = predictMovieForUser(i, user, W, training)

    return predictions



### MAX EPOCH 24 ###
#Training loss = 1.699230
#Validation loss = 1.837046

### MEAN EPOCH 24 ###
#Training loss = 1.123018
#Validation loss = 1.234254

### EPOCH 24 exp ###
#Training loss = 1.164680
#Validation loss = 1.207102

