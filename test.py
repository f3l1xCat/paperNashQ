a = {}
b = []
b.append((1, 2))
b.append((3, 4))
b.append((5, 6))
b.append((7, 8))

a[1] = b

b = []

b.append((9, 10))
b.append((11, 12))
b.append((13, 14))
b.append((15, 16))

a[2] = b

print(a)
print(a[1])
print(a[1][0])
print(a[1][0][0])

c = {}
c[17] = (1, 2)
c[18] = (3, 4)

print(c)
print(c[17])
print(len(c[17]))