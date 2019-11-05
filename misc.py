import numpy as np
import cmath
import os

class Record():
    def __init__(self, FolderName):

        if not os.path.exists(FolderName):
            os.makedirs(FolderName)
        os.chdir(FolderName)
        if not os.path.exists("number_rune.txt"):
            with open("number_rune.txt", "w+") as f:
                f.write("0")
                f.close()

        self.number_run = self.record()
    #
    def record(self):
        print(os.getcwd())
        with open("number_rune.txt", "r") as f:
            a = f.readlines()[0]
            f.close()
        with open("number_rune.txt", "w") as f:
            f.truncate(0)
            f.write(str(int(a)+1))
            f.close()
        if not os.path.exists("run_"+str(int(a)+1)):
            os.makedirs("run_"+str(int(a)+1))
        os.chdir("run_"+str(int(a)+1)) #I leave you in this directory
        # os.chdir("..")
        return int(a)+1



def make_attenuations(layers, how="equal_energy_detected"):
    """"Notice that the methods are the same for L=2, but not for L=3.
        According to Matteo, would be interesting to see if equal_attenuations is better than equal_energy_detected, as the latter is the most used

    """
    if how == "equal_attenuations":

        ats = np.pi*np.ones(layers)/4
        ats[-1] = 0
        return ats
    elif how == "equal_energy_detected":
        if layers == 1:
            return [0]
        else:
            ats=[0]
            for i in range(layers-1):
                ats.append(np.arctan(1/np.cos(ats[i])))
            return np.flip(ats)



class Complex(complex):
    def __repr__(self):
        rp = '%7.5f' % self.real if not self.pureImag() else ''
        ip = '%7.5fj' % self.imag if not self.pureReal() else ''
        conj = '' if (
            self.pureImag() or self.pureReal() or self.imag < 0.0
        ) else '+'
        return '0.0' if (
            self.pureImag() and self.pureReal()
        ) else rp + conj + ip

    def pureImag(self):
        return abs(self.real) < 0.000005

    def pureReal(self):
        return abs(self.imag) < 0.000005


def croots(n):
    if n <= 0:
        return None
    return (Complex(cmath.rect(1, 2 * k * cmath.pi / n)) for k in range(n))





def insert(v,M):
    """
    Takes v, M and returns an array that has, for each element of v, a matrix M

    Example:
    x = [x0,x1]
    y = [[0,0],[0,1],[1,0],[1,1]]
    insert(x,y) returns

    [x0 0 0]
    [x0 0 1]
    [x0 1 0]
    [x0 1 1]
    [x1 0 0]
    [x1 0 1]
    [x1 1 0]
    [x1 1 1]
    """
    try:
        a=M.shape
        if len(a)<2:
            a.append(1)
    except Exception:
         a = [1,len(M)]
    result=np.zeros((a[0]*len(v),a[1] +1 )).astype(int)

    f = len(v)+1
    cucu=0
    for k in v:
        result[cucu:(cucu+a[0]),0] = k
        result[cucu:(cucu+a[0]),1:] = M
        cucu+=a[0]
    return result





def outcomes_universe(L):
    """
    Takes L (# of photodetections in the experiment) and returns
    all possible outcomes in a matrix of 2**L rows by L columns,
    which are all possible sequence of outcomes you can ever get.
    """
    a = np.array([0,1])
    two_outcomes = np.array([[0,0],[0,1],[1,0],[1,1]]).astype(int)
    if L<2:
        return np.array([0,1]).astype(int)
    elif L==2:
        return two_outcomes
    else:
        x = insert(a,two_outcomes)
        for i in range(L-3):
            x = insert(a,x)
        return x.astype(int)


def create_random_complex_list(N):
    list =np.zeros(N, dtype=complex)
    for i in range(N):
        list[i] = np.random.random() + 1j*np.random.random_sample()
    return list
