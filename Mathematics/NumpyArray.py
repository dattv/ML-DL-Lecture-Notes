import numpy as np

# create Array
l = [1., 2., 3.]
a = np.array(l)

print(a)
print(a.shape)
print(a.dtype)

a = np.empty([3, 3])
print(a)

a = np.zeros([3, 5])
print(a)

a = np.ones([4])
print(a)

# Combining Array
a1 = np.array([1, 2, 3])
print(a1)

a2 = np.array([4, 5, 6])
print(a2)

a3 = np.vstack((a1, a2))
print(a3)
print(a3.shape)

a3 = np.hstack((a1, a2))
print(a3)
print(a3.shape)

#One-Dimensional list to array
data = [11, 22, 33, 44, 55]
data = np.array(data)
print(data)
print(type(data))

data = [[11, 22],
        [33, 44],
        [55, 66]]

data = np.array(data)
print(data)
print(type(data))


# One-Dimensional Index
data = np.array([11, 22, 33, 44, 55])
print(data[0])
print(data[4])

# negative index
print(data[-1])
print(data[-5])


A = np.array([[1, 2, 3],
              [1, 2, 3]])
b = 2
print(A + b)