import numpy as np
cimport numpy as np

cpdef fit_ellipse(np.ndarray cont,int s,int r):
    
    # Convert Contours w.r.t X & Y dimensions 
    cdef list q = []
    cdef np.ndarray i
    cdef np.ndarray j
    for i in cont:
        for j in i:
            q.append((j[0],j[1]))
    
    # Get X & Y Array Ready 
    cdef np.ndarray[long, ndim=2] p = np.array(q,dtype=int)
    cdef np.ndarray[long, ndim=1] x_1d = p[:,0]
    cdef np.ndarray[long, ndim=1] y_1d = p[:,1]
    cdef np.ndarray[long, ndim=2] x = x_1d[:,np.newaxis]
    cdef np.ndarray[long, ndim=2] y = y_1d[:,np.newaxis]
    
    # Shape
    cdef int sx, sy
    sx = x.shape[0]
    sy = x.shape[1]
    
    # Prepare Objective Function
    cdef np.ndarray[long, ndim=2] D = np.hstack([x*x,x*y,y*y,x,y,np.ones([sx,sy], dtype=int)])
    cdef np.ndarray[long, ndim=2] S = np.dot(D.T,D)
    cdef np.ndarray[long, ndim=2] C = np.zeros([6,6], dtype=int)
    C[0,2]=C[2,0]=2
    C[1,1]=-1
    
    # Solve Objective for Answer
    cdef np.ndarray[double, ndim=1] E
    cdef np.ndarray[double, ndim=2] V
    E , V = np.linalg.eig(np.dot(np.linalg.inv(S),C))
    cdef int n = np.argmax(E)
    
    # Answer
    cdef np.ndarray[double, ndim=1] an = V[:,n]

    # Fit Ellipse with Answer Found
    cdef float a = an[0]
    cdef float b = an[1]/2.
    cdef float c = an[2]
    cdef float d = an[3]/2.
    cdef float f = an[4]/2.
    cdef float g = an[5]
    cdef float num = b*b-a*c

    # Center Coordinates of Ellipse
    cdef float cx = (c*d-b*f)/num
    cdef float cy = (a*f-b*d)/num

    # Determine Rotation (Interactive Part)
    cdef float angle = 0.5*np.arctan(2*b/(a-c))*r/np.pi
    
    # Determine Axis (Interactive Part)
    cdef float up = s*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    cdef float down1 = (b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    cdef float down2 = (b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    cdef float M = np.sqrt(abs(up/down1))
    cdef float m = np.sqrt(abs(up/down2))

    # Return Solution as Tuple that can be plotted onto Image
    cdef ((float,float),(float,float),float) params
    params = ((cx,cy),(M,m),angle)
    return params