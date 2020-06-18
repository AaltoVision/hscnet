# distutils: language = c++

import numpy as np
cimport numpy as np

from pnpransacpy cimport PnPRANSAC
                     
cdef class pnpransac:
    cdef PnPRANSAC c_pnpransac
    
    def __cinit__(self, float fx, float fy, float cx, float cy):
        self.c_pnpransac = PnPRANSAC(fx, fy, cx, cy)

    def update_camMat(self, float fx, float fy, float cx, float cy):
        self.c_pnpransac.camMatUpdate(fx, fy, cx, cy)

    def RANSAC_loop(self, np.ndarray[double, ndim=2, mode="c"] img_pts, 
                np.ndarray[double, ndim=2, mode="c"] obj_pts, int n_hyp):
        cdef float[:, :] img_pts_ = img_pts.astype(np.float32)
        cdef float[:, :] obj_pts_ = obj_pts.astype(np.float32)
        cdef int n_pts 
        n_pts = img_pts_.shape[0]
        assert img_pts_.shape[0] == obj_pts_.shape[0]
        cdef double* pose
        pose = self.c_pnpransac.RANSACLoop(&img_pts_[0,0], &obj_pts_[0,0],
                n_pts, n_hyp)
        rot =  np.array([pose[0],pose[1],pose[2]])
        transl = np.array([pose[3],pose[4],pose[5]])
        return rot, transl