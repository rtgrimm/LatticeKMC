
import ctypes

import numpy as np
from past.types import long

from Nano import IntVector
from Nano import vector_data


def PtrToArray(ptr, size):
    addr = long(ptr)

    return np.copy(np.ctypeslib.as_array(
        (ctypes.c_int32 * size).from_address(addr)))

def VecToMat(vec : IntVector):
    data = vector_data(vec)
    array = PtrToArray(data.ptr, data.size)

    return array
