
import ctypes

import numpy as np

from Python.Nano import *


def int_ptr_to_array(ptr, size):
    addr = int(ptr)

    return np.copy(np.ctypeslib.as_array(
        (ctypes.c_int32 * size).from_address(addr)))

def int_vec_to_mat(vec : IntVector):
    data = vector_data(vec)
    array = int_ptr_to_array(data.ptr, data.size)

    return array

def IVec3D_to_mat(vec : IVec3DVector):
    data = vector_data(vec)
    array = int_ptr_to_array(data.ptr, data.size)

    return array

def double_ptr_to_array(ptr, size):
    addr = int(ptr)

    return np.copy(np.ctypeslib.as_array(
        (ctypes.c_double * size).from_address(addr)))

def double_vec_to_mat(vec : DoubleVector):
    data = vector_data(vec)
    array = double_ptr_to_array(data.ptr, data.size)

    return array