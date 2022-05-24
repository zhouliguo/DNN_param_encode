import numpy as np
import math
import sys


class ShannonCoding:
    """ 香农编码

        parameters
        ----------

        complement : int
            编码进制

        dim : int
            信源维度

        N : int
            符号个数

        symbols : list
            符号列表

        P : list
            符号取值概率列表

        cumulative_p : lsit
            累加概率列表

        L : list
            码长列表

        code : list
            码字列表

        H : float
            信源熵

        R : float
            信息率

        K : float
            编码效率
            
        ----------

    """

    def __init__(self, symbols, p, complement=2, dim=1):
        p = np.array(p)
        n = len(p)

        if len(symbols) != n:
            print('符号与取值概率个数不匹配!')
            sys.exit(1)

        # 按概率大小排序
        for i in range(n):
            for j in range(n - i - 1):
                if p[j] <= p[j + 1]:
                    p[j], p[j + 1] = p[j + 1], p[j]
                    symbols[j], symbols[j + 1] = symbols[j + 1], symbols[j]

        # 计算累加概率
        cum_p = []
        for i in range(n):
            cum_p.append(0) if i == 0 else cum_p.append(cum_p[i - 1] + p[i - 1])
        cum_p = np.array(cum_p)

        # 计算码长序列
        length = [int(math.ceil(math.log(1 / p[i], complement))) for i in range(n)]

        # 编码
        code = []
        for i in range(n):
            single_code = ''
            t = cum_p[i]
            for j in range(length[i]):
                t = t * complement
                t, z = math.modf(t)
                single_code += str(int(z))
            code.append(single_code)

        hx = np.sum((-1) * np.log2(p) * p)
        r = np.sum(np.array(length) * p) * math.log2(complement) / dim
        k = hx / r

        self.complement = complement    # 编码进制
        self.dim = dim    # 信源维度
        self.N = n     # 符号个数
        self.symbols = symbols    # 符号列表
        self.P = p    # 符号取值概率
        self.cumulative_p = cum_p    # 累加概率
        self.L = length    # 码长列表
        self.code = code    # 码字列表
        self.H = hx    # 信源熵
        self.R = r     # 信息率
        self.K = k     # 编码效率

    def encode(self, img, path='code.txt'):
        """ 编码 """

        c = ''

        img_tmp = np.zeros(img.size, np.int32)
        for i in range(self.N):
            index = np.where(img == self.symbols[i])
            img_tmp[index] = i

        for i in img_tmp:
            c += self.code[i]

        #for point in img:
        #    for i in range(self.N):
        #        if self.symbols[i] == point:
        #            c += self.code[i]

        f = open(path, 'w')
        f.write(c)
        f.close()
        return c

    def decode(self, c):
        """ 解码 """

        a = []
        s = ''
        loc = 0

        '''
        lens = [3,4,5,6,7,8,9,10]

        while c != '':
            print(len(c))
            for l in lens:
                s = c[:l]
                if s in self.code:
                    index = self.code.index(s)
                    a.append(self.symbols[index])
                    c = c[l:]
                    break
        '''

        while c != '':
            s += c[loc]
            loc += 1
            for i in range(self.N):
                if self.code[i] == s:
                    a.append(self.symbols[i])
                    c = c[loc:]
                    loc = 0
                    s = ''
                    break
        

        return np.array(a)

    def print_format(self, describe='Symbols'):
        """ 格式化输出信息 """

        print('{:<10}\t{:<20}\t{:<25}\t{:<10}\t{}'.
              format(describe, 'Probability', 'Cumulative Probability', 'Length', 'Code'))
        print('-' * 100)
        
        if self.N > 15:
            for i in range(5):
                print('{:<10}\t{:<20}\t{:<25}\t{:<10}\t{}'.
                      format(self.symbols[i], self.P[i], self.cumulative_p[i], self.L[i], self.code[i]))
            print('{:<10}\t{:<20}\t{:<25}\t{:<10}\t{}'.
                  format(' ...', ' ...', ' ...', ' ...', ' ...'))
            for i in range(5):
                print('{:<10}\t{:<20}\t{:<25}\t{:<10}\t{}'.
                      format(self.symbols[i-5], self.P[i-5], self.cumulative_p[i-5], self.L[i-5], self.code[i-5]))
        else:
            for i in range(self.N):
                print('{:<10}\t{:<20}\t{:<25}\t{:<10}\t{}'.
                      format(self.symbols[i], self.P[i], self.cumulative_p[i], self.L[i], self.code[i]))

        '''
        for i in range(self.N):
            print('{:<10}\t{:<20}\t{:<25}\t{:<10}\t{}'.
                format(self.symbols[i], self.P[i], self.cumulative_p[i], self.L[i], self.code[i]))
        '''

        print('-' * 100)
