numList = []
for i in range(100, 1000):
    s = str(i)
    if s[0] == s[1] or s[0] == s[2] or s[1] == s[2]:
        continue
    else:
        numList.append(i)

a = 1
while (a!=0):
    ans= str(input("Enter a number:"))
    a , b = input("Enter two numbers(A and B):").split()
    



# a = input()
print(numList)