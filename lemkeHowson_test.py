import lemkeHowson1
import matrix
import rational
import numpy as np
import nashpy

m0 = matrix.Matrix(2, 2)
m1 = matrix.Matrix(2, 2)

m0.setItem(1, 1)
m0.setItem(1, 2)
m0.setItem(2, 1)
m0.setItem(2, 2)

m1.setItem(1, 1)
m1.setItem(1, 2)
m1.setItem(2, 1)
m1.setItem(2, 2)

print(m0)
print(m1)

print(m0.getNumRows())
print(m0.getNumCols())
c1 = m0.getNumRows()
c2 = m0.getNumCols()
r1 = rational.Rational(1, c1)
r2 = rational.Rational(1, c2)
l = [[], []]
for i in range(c1):
    l[0].append(r1)
l[0] = tuple(l[0])
for i in range(c2):
    l[1].append(r2)
l[1] = tuple(l[1])
l = tuple(l)
# print(r)
# probprob = lemkeHowson.lemkeHowson(m0, m1)
probprob = l
print(f"probprob:{probprob}")
print(f"l:{l}")
print(type(probprob))
prob0 = np.array(probprob[0])
prob1 = np.array(probprob[1])
prob0 = np.matrix(prob0)
prob1 = np.matrix(prob1).reshape((-1, 1))
print (f"prob0:{prob0}")
print (f"prob1:{prob1}")
q = []
for i in range(m1.getNumRows()):
    for j in range(m1.getNumCols()):
        q.append(m1.getItem(i+1, j+1))
q = np.matrix(q).reshape((3,2))
prob0 = np.array
print (f"q:{q}")
c = prob0 * q * prob1
print(c)
print (c[0,0].nom() / c[0,0].denom())