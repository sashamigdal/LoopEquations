import numpy as np
import scipy.sparse as sparse
import ctypes

lib_config = "debug" if (os.getenv("USE_DEBUG_LIB") == "1") else "release"
cxx_lib_dir = "CPP/cmake-build-" + lib_config
libDS_path = os.path.join(cxx_lib_dir, 'libDS.so')
sys.path.append(cxx_lib_dir)

def test_ConvertBlockMatrix():
    Z3 = np.zeros((3, 3), dtype=np.int32)
    I3 = np.identity(3, dtype=np.int32)
    mtx = np.empty((4, 4), dtype=np.ndarray)
    mtx.fill(Z3)
    for i in range(4):
        mtx[i][i] = I3
        mtx[i][(i + 1) % 4] = I3

    # mtx = np.array([[I3, I3, Z3, Z3],
    #                 [Z3, I3, I3, Z3],
    #                 [Z3, Z3, I3, I3],
    #                 [I3, Z3, Z3, I3]], dtype=np.ndarray)
    np.set_printoptions(edgeitems=30, linewidth=1000, formatter=dict(object=lambda x: str(x)))
    # print(mtx)

    N = 4
    data = np.empty(2*N, dtype=np.ndarray)
    for i in range(2*N):
        data[i] = I3
    indices = np.array([0, N-1] + [j for i in range(1,N) for j in [i-1, i]])
    indptr = np.array([2 * i for i in range(N+1)])
    csc_mtx = sparse.csc_matrix((data, indices, indptr), shape=(N, N))
    print(csc_mtx)


test_ConvertBlockMatrix()
