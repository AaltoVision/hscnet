from libcpp cimport bool
from libcpp.vector cimport vector

cdef extern from "pnpransac.cpp":
    pass

cdef extern from "pnpransac.h" namespace "poseSolver":
    cdef cppclass PnPRANSAC:
        PnPRANSAC() except +
        PnPRANSAC(float, float, float, float) except +
        void camMatUpdate(float, float, float, float)
        double* RANSACLoop(float*, float*, int, int)