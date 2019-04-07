import numpy as np
import rbm
import projectLib as lib

training = lib.getTrainingData()
validation = lib.getValidationData()
# You could also try with the chapter 4 data
# training = lib.getChapter4Data()

trStats = lib.getUsefulStats(training)
vlStats = lib.getUsefulStats(validation)
print(training)
K = 5
# SET PARAMETERS HERE!!!
# number of hidden units
F = 5
epochs = 40
gradientLearningRate = 0.001

# Initialise all our arrays
W = rbm.getInitialWeights(trStats["n_movies"], F, K)
posprods = np.zeros(W.shape)
negprods = np.zeros(W.shape)
visitingOrder = np.array(trStats["u_users"])
biasOne = np.zeros((trStats["n_movies"], K))

def getRatingsProp(movie, training):
    props = np.zeros((K))
    for x in training:
        if x[0] == movie:
            props += x[3:]
    return props

for i in range(trStats["n_movies"]):
    probs = getRatingsProp(i, training)
    # print(probs)
    probs = probs / np.linalg.norm(probs)
    # print(probs)
    for k in range(K):
        if probs[k] == 1:
            biasOne[i][k] = 0
        if probs[k] != 0:
            biasOne[i][k] = np.log(probs[k]/(1- probs[k]))

for epoch in range(1, epochs+1):
    # in each epoch, we'll visit all users in a random order
    np.random.shuffle(visitingOrder)
    for user in visitingOrder:
        # get the ratings of that user
        ratingsForUser = lib.getRatingsForUser(user, training)
        # build the visible input
        v = rbm.getV(ratingsForUser)
        # get the weights associated to movies the user has seen
        weightsForUser = W[ratingsForUser[:, 0], :, :]
        ### LEARNING ###
        # propagate visible input to hidden units
        posHiddenProb = rbm.visibleToHiddenVec(v, weightsForUser, biasOne)
        # get positive gradient
        # note that we only update the movies that this user has seen!
        posprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(v, posHiddenProb)
        ### UNLEARNING ###
        # sample from hidden distribution
        sampledHidden = rbm.sample(posHiddenProb)
        # propagate back to get "negative data"
        negData = rbm.hiddenToVisible(sampledHidden, weightsForUser)
        # propagate negative data to hidden units
        negHiddenProb = rbm.visibleToHiddenVec(negData, weightsForUser, biasOne)
        # get negative gradient
        # note that we only update the movies that this user has seen!
        negprods[ratingsForUser[:, 0], :, :] += rbm.probProduct(negData, negHiddenProb)

        # we average over the number of users in the batch (if we use mini-batch)
        grad = gradientLearningRate * (posprods - negprods)
        # print(W)
        W = 0.5*W + 0.5*grad
    # Print the current RMSE for training and validation sets
    # this allows you to control for overfitting e.g
    # We predict over the training set
    tr_r_hat = rbm.predict(trStats["movies"], trStats["users"], W, training, b=biasOne)
    trRMSE = lib.rmse(trStats["ratings"], tr_r_hat)

    # We predict over the validation set
    vl_r_hat = rbm.predict(vlStats["movies"], vlStats["users"], W, training, b=biasOne)
    vlRMSE = lib.rmse(vlStats["ratings"], vl_r_hat)

    print "### EPOCH %d ###" % epoch
    print "Training loss = %f" % trRMSE
    print "Validation loss = %f" % vlRMSE
    # break
### END ###
# This part you can write on your own
# you could plot the evolution of the training and validation RMSEs for example

predictedRatings = np.array([rbm.predictForUser(user, W, training, b=biasOne) for user in trStats["u_users"]])
# print(predictedRatings.shape)
np.savetxt("predictedRatingsExp.txt", predictedRatings)
