import numpy as np 

def base_svd (input_matrix,x):
    u,s,v_t= np.linalg.svd(input_matrix)
    u_1= u[:,:x]
    s_1 = s[:x]
    v_t_1 = v_t[:x]
    return u_1,s_1,v_t_1