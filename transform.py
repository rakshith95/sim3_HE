import math
from ast import arg
from asyncio.base_tasks import _task_get_stack
from cmath import inf
from doctest import script_from_examples
from math import degrees
from pathlib import Path

import numpy as np
import rotation_helper_functions as rot_help
import scipy.optimize
from scipy.optimize import Bounds, NonlinearConstraint, least_squares, minimize
from scipy.spatial.transform import Rotation as ROT


class sim3_opt():
    def __init__(self, ref_R, ref_t, trans_R, trans_t):
        self.ref_R = ref_R
        self.ref_t = ref_t
        self.ref_C = np.zeros_like(ref_t)
        self.trans_R = trans_R
        self.trans_t = trans_t
        self.trans_C = np.zeros_like(trans_t)
        self.n_samples = self.ref_R.shape[-1]
        for i in range(self.n_samples):
            self.ref_C[:,i] = -(self.ref_R[:,:,i].T)@self.ref_t[:,i]
            self.trans_C[:,i] = -(self.trans_R[:,:,i].T)@self.trans_t[:,i]

        self.s_init, self.R_init, self.t_init = self.get_initial_estimates()
        self.log_s_init = math.log(self.s_init)
        print("INIT\ns: ",self.s_init,"\nR: \n", self.R_init, "\nt: \n", self.t_init)
        # exit()
        self.R_est = None
        self.t_est = None
        self.s_est = None
        self.log_s_est = None

    def estimate_similarity_transformation(self):
        X_ref = self.ref_C
        X_transformed = self.trans_C
        mc = np.mean(X_transformed, axis=1).reshape((3,1))
        mh = np.mean(X_ref, axis=1).reshape((3,1))
        c0 = X_transformed - mc        
        h0 = X_ref - mh
        normc = np.sqrt(np.sum(np.multiply(c0,c0)))
        normh = np.sqrt(np.sum(np.multiply(h0,h0)))
        c0 = c0 * (1/normc)
        h0 = h0 * (1/normh)

        # find transform. minimize LSQ
        A = h0 @ c0.T
        U, S, V = np.linalg.svd(A)     # numpy uses: A = U * np.diag(S) * V
        R = U @ V

        if np.linalg.det(R) < 0:
            V[2,::] = -V[2,::]
            S[2] = -S[2]
            R = U * V
        s = np.sum(S) * normh / normc
        t = mh - s*R@mc
        return {"scale": s, "rotation": R, "translation": t}        

    def get_initial_estimates(self,R1=None, R2=None, t1=None, t2=None, R=None):
        R_estimates = []
        R1 = self.ref_R if R1 is None else R1
        R2 = self.trans_R if R2 is None else R2
        t1 = self.ref_t if t1 is None else t1
        t2 = self.trans_t if t2 is None else t2

        if R is None:
            for i in range(R1.shape[-1]):
                R1i = R1[:,:,i]
                R2i = R2[:,:,i]
                R_hat_i = R1i.T @ R2i
                R_estimates.append(R_hat_i)

            R_hat = rot_help.geodesicl1mean(R_estimates)
        else:
            R_hat = R

        # R_hat = np.asarray([[-0.68381449,  0.72960981, -0.00820114],
    #    [-0.42349734, -0.40601841, -0.80981421],
    #    [-0.59417821, -0.55028954,  0.58662909]])
        data_mat = None
        res_vec = None
        for i in range(R1.shape[-1]):
            data_mat_element = np.zeros((3,4))
            res_vec_element = np.zeros((3,1))

            data_mat_element[:3,:3] = -1*np.eye(3)
            data_mat_element[:,-1] = R_hat@R2[:,:,i].T@t2[:,i]
            if data_mat is None:
                data_mat = data_mat_element
            else:
                data_mat = np.vstack((data_mat, data_mat_element))

            res_vec_element = R1[:,:,i].T@t1[:,i]
            if res_vec is None:
                res_vec = res_vec_element.reshape((3,1))
            else:
                res_vec = np.vstack((res_vec, res_vec_element.reshape((3,1))))

        '''
        for i in range(R1.shape[-1]):
            data_mat_element = np.zeros((3,4))
            res_vec_element = np.zeros((3,1))

            data_mat_element[:3,:3] = -1*R1[:,:,i]
            data_mat_element[:,-1] = t2[:,i]

            if data_mat is None:
                data_mat = data_mat_element
            else:
                data_mat = np.vstack((data_mat, data_mat_element))

            res_vec_element = t1[:,i]

            if res_vec is None:
                res_vec = res_vec_element.reshape((3,1))
            else:
                res_vec = np.vstack((res_vec, res_vec_element.reshape((3,1))))
        '''

        A = data_mat
        b = res_vec.reshape((-1))
        t_and_s_hat = scipy.optimize.lsq_linear(A, b, bounds=([-np.inf,-np.inf,-np.inf,np.nextafter(0,1)],[np.inf,np.inf,np.inf,np.inf] )).x
        s_hat = t_and_s_hat[-1]
        t_hat = t_and_s_hat[:3]

        return s_hat, R_hat, t_hat

    @staticmethod
    def cons_f(x):
        return [np.sqrt(x[1]**2+x[2]**2+x[3]**2)]

    @staticmethod
    def cons_J(x):
        return [[0, 2*x[1], 2*x[2], 2*x[3],0,0,0]]

    @staticmethod
    def get_skewsymMat(v):
        x = v[0]
        y = v[1]
        z = v[2]

        return np.asarray([ [0,-z,y],[z,0,-x],[-y,x,0] ], dtype=object)

    def callable_fn(self, params, *args):
        log_s,rotvec1, rotvec2, rotvec3, t1,t2,t3 = params
        s = math.exp(log_s)
        rotvec = np.asarray([rotvec1, rotvec2, rotvec3])        
        R = rot_help.vec_to_mat(rotvec)
        u = np.asarray([t1,t2,t3]).reshape((3,1))
        w_hat = sim3_opt.get_skewsymMat(rotvec)
        
        A = np.zeros((4,4))
        A[:3,:3] = log_s*np.eye(3) + w_hat
        A[:3,-1] = u[:,0]        
        T = scipy.linalg.expm(A)
        t = T[:3,-1]
        '''        
        # FIND CLOSED FORM EXP OF SIM(3)
        theta = np.linalg.norm(rotvec)
        if theta != 0:
            V = np.eye(3) + ((1-np.cos(theta))/theta**2)*w_hat + ((theta - np.sin(theta))/(theta**3))*np.linalg.matrix_power(w_hat,2)
        else:
            V = np.eye(3)
 
        u = V@t.reshape((3,1))
        T2 = np.zeros((4,4))
        T2[:3,:3] = s*R
        T2[:3,-1] = u[:,0]
        T2[-1,-1] = 1
        '''

        ref_R = args[0]
        ref_t = args[1]

        trans_R = args[2]
        trans_t = args[3]
        # w_rot, w_trans = args[4]
        residuals = []

        for i in range(ref_R.shape[-1]):
            R1 = ref_R[:,:,i]
            t1 = ref_t[:,i].reshape((3,1))

            R2 = trans_R[:,:,i]
            t2 = trans_t[:,i].reshape((3,1))

            T1 = np.zeros((4,4))
            T1[:3,:3] = R1
            T1[:3,-1] = t1[:,0]
            T1[-1,-1] = 1

            T2 = np.zeros((4,4))
            T2[:3,:3] = R2
            T2[:3,-1] = t2[:,0]
            T2[-1,-1] = 1            
            
            T_scale = np.zeros((4,4))
            T_scale[:3,:3] = s*np.eye(3)
            T_scale[-1, -1] = 1

            T2_hat = np.linalg.inv(T_scale)@T1@T 
            resid_transform = T2_hat@np.linalg.inv(T2)
            # resid_transform = np.linalg.inv(T2)@np.linalg.inv(T_scale)@T1@T
            # resid_transform = np.linalg.inv(T1)@T_scale@T2@np.linalg.inv(T)
            resid_scaled_R_mat = resid_transform[:3,:3]
            resid_s = np.linalg.det(resid_scaled_R_mat)**(1. / 3) - 1
            resid_R_mat = (1/(resid_s+1))*resid_scaled_R_mat
            # print(np.linalg.det(resid_R_mat))
            resid_rvec = rot_help.mat_to_vec(resid_R_mat).reshape((3,1))
            theta = np.linalg.norm(resid_rvec)
            resid_w_hat = sim3_opt.get_skewsymMat(resid_rvec)
            if theta !=0:
                V_inv = np.eye(3) - (1/2)*resid_w_hat +  ((1 - ((theta*np.cos(theta/2))/(2*np.sin(theta/2))) )/theta**2)*np.linalg.matrix_power(resid_w_hat,2)
            else:
                V_inv = np.eye(3)
            resid_t = V_inv@resid_transform[:-1,-1].reshape((3,1))

            '''
            resid_R = resid_transform[:3,:3]
            resid_rvec = rot_help.mat_to_vec(resid_R).reshape((3,1))
            log_resid_transform = scipy.linalg.logm(resid_transform)
            resid_t = log_resid_transform[:3,-1].reshape((3,1))
            resid_s = np.linalg.det(resid_R)**(1. / 3) - 1
            # resid_t = resid_transform[:3,-1].reshape((3,1))

            resid_log_s = math.log(np.linalg.det(resid_R)**(1. / 3)) - math.log(1)

            # theta = np.linalg.norm(resid_rvec)
            '''
            # trans_vec = t - s*R@R2.T@t2.reshape((3,1)) + (R1.T@t1).reshape((3,1))
            # resid = resid_rvec
            resid = np.vstack(( resid_rvec, resid_t, resid_s ))
            e = np.linalg.norm(resid)

            residuals.append(e)

        residuals = np.asarray(residuals)
        e = np.sum(residuals)#np.asarray(np.linalg.norm(residuals, ord=1)).reshape((1,1))
        return e


    def estimate_RANSAC(self, p_succ=0.99, p_inliers=0.9, err_thresh =1.5, iterations=None):
        minimal_pairs = 2
        n_samples = self.n_samples
        indices = np.arange(0, n_samples)
        nIterations = round(math.log((1-p_succ))/math.log(1-p_inliers**minimal_pairs)) if iterations is None else iterations
        nbest = 0
        best = None
        best_inliers = None
        for i in range(nIterations):
            sample_indices = np.random.choice(indices, minimal_pairs)
            if sample_indices[0] == sample_indices[1]:
                continue
            R1 = self.ref_R[:,:, sample_indices]
            R2 = self.trans_R[:,:, sample_indices]

            t1 = self.ref_t[:,sample_indices].reshape((3,minimal_pairs))
            t2 = self.trans_t[:,sample_indices].reshape((3,minimal_pairs))
            s, R, t = self.get_initial_estimates(R1, R2, t1, t2)

            T = np.zeros((4,4))
            T[:3,:3] = s*R
            T[:3,-1] = t[:]
            T[-1,-1] = 1
            residuals = []
            for i in range(n_samples):
                R1i = self.ref_R[:,:,i]
                R2i = self.trans_R[:,:,i]

                t1i = self.ref_t[:,i].reshape((3,1))
                t2i = self.trans_t[:,i].reshape((3,1))
    
                T1 = np.zeros((4,4))
                T1[:3,:3] = R1i
                T1[:3,-1] = t1i[:,0]
                T1[-1,-1] = 1
    
                T2 = np.zeros((4,4))
                T2[:3,:3] = R2i
                T2[:3,-1] = t2i[:,0]
                T2[-1,-1] = 1
    
                T_scale = np.zeros((4,4))
                T_scale[:3,:3] = s*np.eye(3)
                T_scale[-1, -1] = 1
    
                resid_transform = np.linalg.inv(T1)@T_scale@T2@np.linalg.inv(T)
                resid_R_mat = resid_transform[:3,:3]
                resid_s = np.linalg.det(resid_R_mat)**(1. / 3) - 1
                resid_R_mat = 1/(resid_s+1)*resid_R_mat
                resid_rvec = rot_help.mat_to_vec(resid_R_mat).reshape((3,1))
                theta = np.linalg.norm(resid_rvec)
                resid_w_hat = sim3_opt.get_skewsymMat(resid_rvec)
                if theta !=0:
                    V_inv = np.eye(3) - (1/2)*resid_w_hat +  ((1 - ((theta*np.cos(theta/2))/(2*np.sin(theta/2))) )/theta**2)*np.linalg.matrix_power(resid_w_hat,2)
                else:
                    V_inv = np.eye(3)
                resid_t = V_inv@resid_transform[:-1,-1].reshape((3,1))

                resid = np.vstack((resid_rvec, resid_t, resid_s))
                e = np.linalg.norm(resid)
                residuals.append(e)
            residuals = np.asarray(residuals)
            inlier_indices = np.where(residuals<err_thresh)[0]
            num_inliers = len(inlier_indices)
            if num_inliers >= nbest:
                nbest = num_inliers
                best = (s,R,t)
                best_inliers = inlier_indices
        
        s_best, R_best, t_best = best
        
        ref_R_inliers = self.ref_R[:,:, best_inliers]
        ref_t_inliers = self.ref_t[:, best_inliers]
        trans_R_inliers = self.trans_R[:,:,best_inliers]
        trans_t_inliers = self.trans_t[:, best_inliers]
        
        inliers_tuple = (ref_R_inliers, ref_t_inliers, trans_R_inliers, trans_t_inliers)
        return best, inliers_tuple  

    def minimization(self, ref_Rs=None, ref_ts=None, trans_Rs=None, trans_ts=None, log_s_init=None, R_init=None, t_init=None):
        R_refs = self.ref_R if ref_Rs is None else ref_Rs
        t_refs = self.ref_t if ref_ts is None else ref_ts
        R_trans = self.trans_R if trans_Rs is None else trans_Rs
        t_trans = self.trans_t if trans_ts is None else trans_ts
        
        log_s_hat = self.log_s_init if log_s_init is None else log_s_init
        R_hat = self.R_init if R_init is None else R_init
        t_hat = self.t_init if t_init is None else t_init
    
        rvec_hat = rot_help.mat_to_vec(R_hat)

        # OPt problem
        bounds = Bounds([math.log(np.nextafter(0,1)),-np.inf,-np.inf,-np.inf,-np.inf,-np.inf,-np.inf],[np.inf,np.inf,np.inf,np.inf,np.inf,np.inf,np.inf])
        from scipy.optimize import BFGS
        nonlin_constraint = NonlinearConstraint(sim3_opt.cons_f, 0, np.pi, jac=sim3_opt.cons_J, hess=BFGS())
        x0 = [log_s_hat, rvec_hat[0], rvec_hat[1], rvec_hat[2], t_hat[0], t_hat[1], t_hat[2]]
        res = minimize(self.callable_fn, x0, args=(R_refs, t_refs, R_trans, t_trans), jac='3-point', constraints=[nonlin_constraint], bounds=bounds
        , method='trust-constr')
    
        # self.s_est = res.x[0]
        self.log_s_est = res.x[0]
        self.s_est = math.exp(self.log_s_est)
        self.R_est = rot_help.vec_to_mat(np.asarray([res.x[1], res.x[2], res.x[3] ]))
        self.t_est = np.asarray([res.x[4], res.x[5], res.x[6]]).reshape((3,1))
        
        A_opt = np.zeros((4,4))
        A_opt[:3,:3] = self.log_s_est*np.eye(3) + sim3_opt.get_skewsymMat(np.asarray([ res.x[1], res.x[2], res.x[3] ]))
        A_opt[:3,-1] = self.t_est[:,0]
        
        T_opt = scipy.linalg.expm(A_opt)

        opt_sR = T_opt[:3,:3]
        s_ = np.linalg.det(opt_sR)**(1. / 3) 
        R_est = 1/s_*opt_sR
        opt_t = T_opt[:3,-1]
        print( "OPT \ns: ", s_ , '\n\nR:\n', R_est,'\t\t','\n\nt: ', opt_t)
        return (s_,R_est,opt_t)

    
if __name__ == '__main__':
    ref_R = np.loadtxt('/home/rakshith/CTU/ARI/ref_R_reshaped.txt')
    ref_R = ref_R.reshape((3,3,int(ref_R.shape[-1]/3)))
    ref_C = np.loadtxt('/home/rakshith/CTU/ARI/ref_C.txt')

    trans_R = np.loadtxt('/home/rakshith/CTU/ARI/trans_R_reshaped.txt')
    trans_R = trans_R.reshape((3,3,int(trans_R.shape[-1]/3)))
    trans_C = np.loadtxt('/home/rakshith/CTU/ARI/trans_C.txt')

    ref_t = np.array([]).reshape(3,0)
    trans_t = np.array([]).reshape(3,0)

    for i in range(ref_R.shape[-1]):
        Ri = ref_R[:,:,i]
        Ci = ref_C[:,i].reshape((3,1))
        ti = (-Ri@Ci).reshape((3,1))
        ref_t = np.concatenate((ref_t, ti), axis=1)
    
        R_pi = trans_R[:,:,i]
        C_pi = trans_C[:,i].reshape((3,1))
        t_pi = (-R_pi@C_pi).reshape((3,1))
        trans_t = np.concatenate((trans_t, t_pi), axis=1)

    '''
    Rtmp1 = ROT.random().as_matrix()
    Rtmp2 = ROT.random().as_matrix()
    Rtmp3 = ROT.random().as_matrix()
    Rtmp4 = ROT.random().as_matrix()

    tmpt1 = np.random.random((3))
    tmpt2 = np.random.random((3))
    tmpt3 = np.random.random((3))
    tmpt4 = np.random.random((3))

    R_rand = ROT.random().as_matrix()
    R_rand2 = ROT.random().as_matrix()
    t_rand = np.random.random((3,1))
    s_rand = np.random.random(1)[0]

    T_rand = np.zeros((4,4))
    T_rand[:3,:3] = s_rand*R_rand
    T_rand[:3,-1] = t_rand[:,0]
    T_rand[-1,-1] = 1
    
    print("ORIG\n",s_rand,'\n', R_rand,'\n\n', t_rand)
    R2tmp1 = Rtmp1@R_rand
    t2tmp1 = (1/s_rand)*(R2tmp1@R_rand.T@t_rand).reshape((3,1)) + (1/s_rand)*(R2tmp1@R_rand.T@Rtmp1.T@tmpt1).reshape((3,1))
    t2_TEST1 = R_rand2@t2tmp1

    R2tmp2 = Rtmp2@R_rand
    R2_TEST2 = R_rand2@R2tmp2
    t2tmp2 = (1/s_rand)*(R2tmp2@R_rand.T@t_rand).reshape((3,1)) + (1/s_rand)*(R2tmp2@R_rand.T@Rtmp2.T@tmpt2).reshape((3,1))
    t2_TEST2 = R_rand2@t2tmp2
    
    R2tmp3 = Rtmp3@R_rand
    R2_TEST3 = R_rand2@R2tmp3
    t2tmp3 = (1/s_rand)*(R2tmp3@R_rand.T@t_rand).reshape((3,1)) + (1/s_rand)*(R2tmp3@R_rand.T@Rtmp3.T@tmpt3).reshape((3,1))
    t2_TEST3 = R_rand2@t2tmp3

    R2_TEST1 = R_rand2@R2tmp1
    R2tmp4 = Rtmp4@R_rand
    R2_TEST4 = R_rand2@R2tmp4
    t2tmp4 = (1/s_rand)*(R2tmp4@R_rand.T@t_rand).reshape((3,1)) + (1/s_rand)*(R2tmp4@R_rand.T@Rtmp4.T@tmpt4).reshape((3,1))
    t2_TEST4 = R_rand2@t2tmp4

    ref_R = np.stack((Rtmp1, Rtmp2, Rtmp3, Rtmp4), axis=-1)
    trans_R = np.stack((R2tmp1, R2tmp2, R2tmp3, R2tmp4), axis=-1)
    TEST_R = np.stack((R2_TEST1, R2_TEST2, R2_TEST3, R2_TEST4), axis=-1)

    ref_t = np.stack((tmpt1, tmpt2, tmpt3, tmpt4), axis=-1).reshape((3,-1))
    trans_t = np.stack((t2tmp1, t2tmp2, t2tmp3, t2tmp4), axis=-1).reshape((3,-1))
    TEST_t = np.stack((t2_TEST1, t2_TEST2, t2_TEST3, t2_TEST4), axis=-1).reshape((3,-1))
    '''

    optimizer = sim3_opt(ref_R, ref_t, trans_R, trans_t)
    inits, pairs = optimizer.estimate_RANSAC(err_thresh=1,iterations=300)
    s_init_ransac, R_init_ransac, t_init_ransac = inits 
    ref_R_inliers, ref_t_inliers, trans_R_inliers, trans_t_inliers = pairs
    s_init_separable,R_init_separable,t_init_separable = optimizer.get_initial_estimates(R1 = ref_R_inliers, R2 = trans_R_inliers, t1 = ref_t_inliers, t2 = trans_t_inliers)
    print("\nRANSAC+CLOSED-FORM\ns: ",s_init_separable,"\nR: \n", R_init_separable, "\nt: \n", t_init_separable)

    svd_dict = optimizer.estimate_similarity_transformation()
    s,R,t = optimizer.minimization(ref_Rs=ref_R_inliers, ref_ts=ref_t_inliers, trans_Rs=trans_R_inliers, trans_ts=trans_t_inliers, log_s_init=math.log(s_init_separable), R_init = R_init_separable, t_init= t_init_separable)
    s,R,t = optimizer.get_initial_estimates(R1 = ref_R_inliers, R2 = trans_R_inliers, t1 = ref_t_inliers, t2 = trans_t_inliers, R=R)
    print("\FINAL FORM\ns: ",s,"\nR: \n", R, "\nt: \n", t)
    
    '''
    optimizer2 = sim3_opt(trans_R, trans_t, TEST_R, TEST_t)
    s,R,t = optimizer2.minimization(log_s_init=math.log(1), R_init = np.eye(3), t_init=np.zeros((3,1)))

    T2 = np.zeros_like(T_rand)
    T2[:3,:3] = R2tmp1#R_rand2
    T2[:3,-1] = t2tmp1.reshape((3,))
    T2[-1,-1] = 1
    
    Tr = np.zeros_like(T2)
    Tr[:3,:3] = R_rand2
    Tr[-1,-1] = 1
    print('\n\n', np.linalg.inv(T2)@Tr@T2)


    T2 = np.zeros_like(T_rand)
    T2[:3,:3] = R2tmp2#R_rand2
    T2[:3,-1] = t2tmp2.reshape((3,))
    T2[-1,-1] = 1
    print('\n\n', np.linalg.inv(T2)@Tr@T2)



    svd_dict2 = optimizer2.estimate_similarity_transformation()
    '''
    print('\n\n', svd_dict )