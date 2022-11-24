import numpy as np 
import math 
from scipy.spatial.transform import Rotation as ROT


def mat_to_vec(R):
    rot = ROT.from_matrix(R)
    rot_vec = rot.as_rotvec()
    return rot_vec

def vec_to_mat(r):
    rot = ROT.from_rotvec(r)
    R = rot.as_matrix()
    return R

def vec_to_mat(r):
    rot = ROT.from_rotvec(r)
    R = rot.as_matrix()
    return R


def logarithm_map(input_mat):
    cos_theta = (np.trace(input_mat)-1)/2
    if cos_theta > 1 and (cos_theta - 1) <= 0.0001:
        cos_theta = 1.0
    elif cos_theta > 1:
        print(cos_theta - 1, "GG not working")
        exit()
    theta = np.arccos(cos_theta)
    sin_theta = np.sqrt(1-cos_theta**2)
    if sin_theta == 0:
        return np.asarray([0,0,0])
    ln_R = theta/(2*sin_theta)*(input_mat-input_mat.T)
    return np.asarray([ln_R[2,1],ln_R[0,2],ln_R[1,0]])

def geodesicl1mean(R_input, b_outlier_rejection=True, n_iterations=10, thr_convergence=0.001):
    n_samples = len(R_input)
    R_vectors = np.asarray([R.flatten() for R in R_input])
    s = np.median(R_vectors, axis=0)
    U,_ ,VT = np.linalg.svd(s.reshape((3,3)))
    V = VT.T
    R = U@VT
    if np.linalg.det(R) < 0:
        V[:,-1] = -V[:,-1]
        R = U@V.T

    # optimize
    for i in range(n_iterations):
        # for z in range(n_samples):
            # print(R_input[z], R.T)
        vs = [logarithm_map(R_in@R.T) for R_in in R_input]
        
        v_norms = [np.linalg.norm(v) for v in vs]
        thr = np.inf
        if b_outlier_rejection:
            sorted_v_norms = sorted(v_norms)
            v_norm_firstQ = sorted_v_norms[math.ceil(n_samples/4)]
            if n_samples <= 50:
                thr = max(v_norm_firstQ, 1)
            else:
                thr = max(v_norm_firstQ, 0.5)
        step_num = 0
        step_den = 0
        for j in range(n_samples):
            v = vs[j]
            v_norm = v_norms[j]
            if v_norm > thr or v_norm == 0:
                continue
            step_num = step_num + v/v_norm
            step_den = step_den + 1/v_norm
       
        if step_den == 0:
            delta=np.asarray([0,0,0])
        else:
            delta = step_num/step_den
        delta_angle = np.linalg.norm(delta)
        # delta_axis = delta/delta_angle
        R_delta = vec_to_mat(delta)
        R = R_delta@R

        if delta_angle < thr_convergence:
            break
    
    return R