a = [1,2,3,4,5]
b = [0]*3
b[:len(a)] = a
print(b)