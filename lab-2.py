# import numpy as np
#
# # Tutorial: https://www.coursera.org/workspaces/authenticate/tdonnvwheucsaztgsmayfr?forceRefresh=false&next=/notebooks/numpy_tutorial.ipynb
#
# def get_lenght(X):
#     X_T = np.transpose(X)
#     value = X_T @ X
#     length = np.sqrt(value)
#
#     return length
#
#
# def get_angle(X,Y):
#     pass
#
# def get_dot_product(X,Y):
#     return np.transpose(X) @ Y
#
#
# def get_inner_different_product(X,A,Y):
#     return np.transpose(X)@A@Y
#
#
# def distance(x0, x1):
#     """Compute distance between two vectors x0, x1 using the dot product"""
#     x0_x1 = x0 - x1
#     dot_product =  np.transpose(x0_x1) @ x0_x1
#
#     distance = np.sqrt(dot_product)
#     return distance
#
# def angle(x0, x1):
#     dot_product = np.transpose(x0) @ x1
#     x0_value = np.transpose(x0) @ x0
#     x1_value = np.transpose(x1) @ x1
#
#     result = dot_product/(np.sqrt(x0_value) * np.sqrt(x1_value))
#
#     angle = np.arccos(result) # <-- EDIT THIS to compute angle between x0 and x1
#     return angle
#
# X = np.array([[1],[2]])
# Y= np.array([[4],[5]])
# A= np.array([[1, 0 ],[-5, 0]])
#
#
# a = np.array([1,0])
# b = np.array([-1,0])
# print(angle(a, b))

# GRADED FUNCTION: DO NOT EDIT THIS LINE
# def most_similar_image():
#     """Find the index of the digit, among all MNIST digits
#        that is the second-closest to the first image in the dataset (the first image is closest to itself trivially).
#        Your answer should be a single integer.
#     """
#     distances = [[0,0],[1,5],[2,9],[3,0]]
#
#
#     result = sorted(distances, key=lambda x: x[1])
#
#     index = -1
#     for i in range(0,len(result)):
#         if(result[i][1] != 0):
#             index = result[i][0]
#             break
#
#     return index
#
# most_similar_image()


# GRADED FUNCTION: DO NOT EDIT THIS LINE

import numpy as np

def pairwise_distance_matrix(X, Y):
    """Compute the pairwise distance between rows of X and rows of Y

    Arguments
    ----------
    X: ndarray of size (N, D)
    Y: ndarray of size (M, D)

    Returns
    --------
    distance_matrix: matrix of shape (N, M), each entry distance_matrix[i,j] is the distance between
    ith row of X and the jth row of Y (we use the dot product to compute the distance).
    """

    N, D = X.shape
    M = Y.shape[0]
    distance_matrix = np.zeros((N, M)) # <-- EDIT THIS to compute the correct distance matrix.

    i = 0
    for rowX in X:
        j = 0
        for rowY in Y:
            x_y = np.subtract(rowX, rowY)
            dot_product = np.transpose(x_y) @ x_y
            distance = np.sqrt(dot_product)
            distance_matrix[i,j] = distance
            j = j + 1
        i = i + 1


a = np.array([[1,2],[2,2],[3,2]])
b = np.array([[3,3],[4,3],[5,3],[10,3]])
pairwise_distance_matrix(a, b)















