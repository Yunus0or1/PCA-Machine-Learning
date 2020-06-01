import numpy as np

# GRADED FUNCTION: DO NOT EDIT THIS LINE

def normalize(X):
    """Normalize the given dataset X
    Args:
        X: ndarray, dataset

    Returns:
        (Xbar, mean, std): tuple of ndarray, Xbar is the normalized dataset
        with mean 0 and standard deviation 1; mean and std are the
        mean and standard deviation respectively.

    Note:
        You will encounter dimensions where the standard deviation is
        zero, for those when you do normalization the normalized data
        will be NaN. Handle this by setting using `std = 1` for those
        dimensions when doing normalization.
    """
    mu = np.mean(X)  # <-- EDIT THIS, compute the mean of X
    std = np.std(X, axis=0)
    std_filled = std.copy()
    std_filled[std == 0] = 1.
    Xbar = (X-mu) / std  # <-- EDIT THIS, compute the normalized data Xbar
    return Xbar, mu, std    # <-- EDIT THIS, compute the normalized data Xbar

def eig(S):



    eigvals, eigvecs = np.linalg.eig(S)  # compute eigenvectors and eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]
    return (eigvals, eigvecs)


def projection_matrix(B):


    P = B @ np.linalg.inv(np.transpose(B) @ B) @ np.transpose(B)
    return P

def PCA(X, num_components):

    S = np.cov(X, rowvar=False, bias=True)
    eigvals, eigvecs = np.linalg.eig(S)  # compute eigenvectors and eigenvalues
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:,idx]

    print(eigvecs)
    B = eigvecs[:, :num_components]

    print(B)

    X_reconstruct = (projection_matrix(B) @ X.T).T

    return X_reconstruct

# GRADED FUNCTION: DO NOT EDIT THIS LINE
### PCA for high dimensional datasets

def PCA_high_dim(X, n_components):
    """Compute PCA for small sample size but high-dimensional features.
    Args:
        X: ndarray of size (N, D), where D is the dimension of the sample,
           and N is the number of samples
        num_components: the number of principal components to use.
    Returns:
        X_reconstruct: (N, D) ndarray. the reconstruction
        of X from the first `num_components` pricipal components.
    """
    return X # <-- EDIT THIS to return the reconstruction of X

x = np.array([[1,2,3],[2,9,6],[3,4,10]])
PCA_high_dim(x,2)


















