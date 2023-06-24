import numpy as np

def IsSorted(a):
    return (np.diff(a) >=0).all()

class SortedArrayIter():
    """
    Iterates over sorted array of data returning intervals of constant values.
    For example, [0,0,0,1,1,1,2,2,3,3] -> (0,3) (3,6) (6,8) (8,10)
    """
    def __init__(self, a):
        assert( a.ndim ==1)
        b = np.diff(a)
        assert( (b >=0).all())
        nz = np.nonzero(b)[0] + 1
        self.nz = np.zeros(len(nz) +1,nz.dtype)
        self.nz[1:] = nz
        #these nz are indexes of unique ordered values of r
        self.nnz = len(a)

    def __iter__(self):
        for i in range(len(self.nz)):
            yield  self.nz[i], self.nz[i + 1] if i + 1 < len(self.nz) else self.nnz


def  test_Iter():
    a = np.array([0, 1, 2, 2, 2, 3, 4, 4, 5])
    deltas = []
    for delta in SortedArrayIter(a):
        deltas.append(delta)
    assert( deltas == [(0, 1), (1, 2), (2, 5), (5, 6), (6, 8), (8, 9)])
