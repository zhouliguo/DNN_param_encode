#Exponential-Golomb coding
def exp_golomb_code(x, sign=True):
    if sign:
        if x == 0:
            return '1'
        if x > 0:
            x = 2*x-1
        else:
            x = -2*x
            
        x_a = x+1
        x_a_bin = bin(x_a).replace('0b','')
        z = x_a_bin.replace('1','0')[:-1]
        g = z+x_a_bin
    else:
        if x == 0:
            return '1'     
        x_a = x+1
        x_a_bin = bin(x_a).replace('0b','')
        z = x_a_bin.replace('1','0')[:-1]
        g = z+x_a_bin

    return g