import math

def Hex2Bin(input):
    try:
        n1, n2 = '{0:.10f}'.format(input).split('.')
    except ValueError:
        return 'Failed'
    s1 = bin(int(n1))
    s2 = bin(int(n2))[2:]
    if s1[0]=='-':
        s1 = '-'+s1[3:]
    else:
        s1 = s1[2:]
    return s1+'.'+s2

def Bin2Hex(input):
    s1, s2 = input.split('.')
    '''
    if s1=='b1':
        print(input)
    '''
    n1 = int(s1, 2)
    n2 = int(s2, 2)
    while n2>1:
        n2 /= 10
    return n1+n2

if __name__ == '__main__':
    print(Hex2Bin(2.1))
    print(Bin2Hex('10.1'))

# Source: https://stackoverflow.com/questions/4838994/float-to-binary?rq=1