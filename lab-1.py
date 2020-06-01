
import numpy as np






def cov(X):
    "Compute the covariance for a dataset"
    # X is of size (D,N)
    # It is possible to vectorize our code for computing the covariance with matrix multiplications,
    # i.e., we do not need to explicitly
    # iterate over the entire dataset as looping in Python tends to be slow
    # We challenge you to give a vectorized implementation without using np.cov, but if you choose to use np.cov,
    # be sure to pass in bias=True.
    D, N = X.shape
    ### Edit the code to compute the covariance matrix
    covariance_matrix = np.zeros((D, D))
    ### Update covariance_matrix here

    ###
    return covariance_matrix


def mean_naive(X):
    "Compute the mean for a dataset X nby iterating over the data points"
    # X is of size (D,N) where D is the dimensionality and N the number of data points
    D, N = X.shape
    ### Edit the code to compute a (D,1) array `mean` for the mean of dataset.
    mean = np.zeros((D, 1))
    k = 0
    for i in range(0,D):
        sum = 0
        for j in range(0,N):
            sum = sum + X[i][j]
        mean[k] = sum / N
        k  = k+1
    return mean

def mean(X):
    "Compute the mean for a dataset of size (D,N) where D is the dimension and N is the number of data points"
    # given a dataset of size (D, N), the mean should be an array of size (D,1)
    # you can use np.mean, but pay close attention to the shape of the mean vector you are returning.

    D, N = X.shape
    ### Edit the code to compute a (D,1) array `mean` for the mean of dataset.

    mean = np.mean(X,axis=1).reshape(D,1)

    return mean

def cov_naive(X):
    """Compute the covariance for a dataset of size (D,N)
    where D is the dimension and N is the number of data points"""
    D, N = X.shape
    ### Edit the code below to compute the covariance matrix by iterating over the dataset.
    covariance = np.zeros((D, D))
    covMatrix = np.cov(X,bias=True)
    print(covMatrix)
    ### Update covariance

    ###
    return covMatrix


# Let's first test the functions on some hand-crafted dataset.

X_test = np.arange(6).reshape(2,3)
expected_test_mean = np.array([1., 4.]).reshape(-1, 1)
expected_test_cov = np.array([[2/3., 2/3.], [2/3.,2/3.]])
mean(X_test)

print('X:\n', X_test)
print('Expected mean:\n', expected_test_mean)
print('Expected covariance:\n', expected_test_cov)


print("My Result")

print(mean(X_test))
# print(np.testing.assert_almost_equal(mean(X_test), expected_test_mean))
# print(np.testing.assert_almost_equal(mean_naive(X_test), expected_test_mean))
#
# print(np.testing.assert_almost_equal(cov(X_test), expected_test_cov))
# np.testing.assert_almost_equal(cov_naive(X_test), expected_test_cov)