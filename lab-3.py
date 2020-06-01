
import numpy as np

def projection_1d(b):

    dot_product1 = b @ np.transpose(b)
    dot_product2 =  np.transpose(b) @  b


    return dot_product1/dot_product2



def get_lambda(b,x):

    dot_product1 = np.transpose(b)@ x
    dot_product2 = np.transpose(b) @ b

    return dot_product1/dot_product2

def distance(x0, x1):
    """Compute distance between two vectors x0, x1 using the dot product"""
    x0_x1 = x0 - x1
    dot_product =  np.transpose(x0_x1) @ x0_x1

    distance = np.sqrt(dot_product)
    return distance




def get2D_projection_matrix(B):
    return B @ np.linalg.inv(np.transpose(B) @ B) @ np.transpose(B)

def get2D_lambda(B,x):
    return np.linalg.inv(np.transpose(B) @ B) @ np.transpose(B) @ x

def get2D_projection_point(B,x):
    lambda_value = get2D_lambda(B,x)
    return B @ lambda_value

b1 = np.array([[1],[1],[1]])
b2 = np.array([[0],[1],[2]])
x = np.array([[12],[0],[0]])
B = np.concatenate((b1,b2),axis=1)

print(get_lambda(get2D_projection_point(B,x)))

