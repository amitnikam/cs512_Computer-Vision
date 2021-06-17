import numpy as np
cimport numpy as np
DTYPE = np.int
ctypedef np.int_t DTYPE_t

cpdef np.ndarray smoothing(int n, np.ndarray image_instance_S):
    cdef int dn = 1 + (2*n)
    cdef np.ndarray kernel = np.ones((dn,dn),np.float32)/(dn*dn)
    cdef np.ndarray output = np.zeros_like(image_instance_S)
    cdef np.ndarray image_padded = np.zeros((image_instance_S.shape[0] + (dn-1), image_instance_S.shape[1] + (dn-1)))
    image_padded[n:-n, n:-n] = image_instance_S
    cdef int x
    cdef int y
    for x in range(image_instance_S.shape[1]):
        for y in range(image_instance_S.shape[0]):
            # element-wise multiplication of the kernel and the image
            output[y, x] = (kernel * image_padded[y:y+dn, x:x+dn]).sum()
    return output