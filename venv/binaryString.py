
#from string to binary
def a2bits(a):
    return ''.join(format(ord(i), '08b') for i in str(a))

#from binary to string
def bits2a(b):
    return ''.join(chr(int(''.join(x), 2)) for x in zip(*[iter(b)]*8))




