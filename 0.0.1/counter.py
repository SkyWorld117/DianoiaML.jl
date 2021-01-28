s = []
for i in range(10):
    for j in range(10):
        if (i+j in s) == False:
            s.append(i+j)
        if (i-j in s) == False:
            s.append(i-j)
        if (i*j in s) == False:
            s.append(i*j)
        if j != 0:
            if i//j in s == False:
                s.append(i//j)
s.append(None)
#print('length:', len(s)+1)
#print('max:', max(s))
#print('min:', min(s))
#print(s)
