import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import sqrtm

class BGL_MODEL:

    def __init__(self, postBG_target_range=(70,180), postBG_target_value=112.5, variables=None):

        self.postBG_target_range = postBG_target_range
        self.postBG_target_value = postBG_target_value
        self.variables = variables  

        self.X_LTS = None  # action array 
        self.m_LTS = None  # measurement array


    def recommend(self, context=None, ins_calc=None, strategy=None, aux_ins=None, ins_ceiling=None, S=50, L=200, lam=2, exp_coeff=0.1):

        ins_space =  np.round(np.linspace(0, ins_ceiling, ins_ceiling*aux_ins + 1), 3)

        if strategy=='salelts':
            ins = self._acqu_with_salelts(context, ins_space, ins_calc=ins_calc, S=S, L=L, lam=lam, exp_coeff=exp_coeff)
        elif strategy=='lelts':
            ins = self._acqu_with_lelts(context, ins_space, S=S, L=L, lam=lam, exp_coeff=exp_coeff)
        else:
            ins = -1

        return ins


    def _acqu_with_salelts(self, context, ins_feasible, ins_calc, S, L, lam, exp_coeff):
        
        #print('+-+- SALE-LTS -+-+') 
        target_bg = self.postBG_target_value
        bg_omin, bg_omax = self.postBG_target_range[0], self.postBG_target_range[1]  
        R = 5  # the measurement noise should be noise is R-sub-gaussian for example if the noise is gaussian its variance must be at most R^2
        d = len(self.variables)
        t = 0 if self.X_LTS is None else self.X_LTS.shape[0]  # round (rec.) number
        lam = lam  # regularizaton constant for ridge regression
        delta = 1e-12  # with at least 1 - delta probability.. 
        S = S  # maximum value of one element of theta.
        L = L  # max val of context elements
        ins_normalized = ins_feasible
        safe_norm_ins = min(ins_feasible, key=lambda x: abs(x - ins_calc))
    
        Vt = lam*np.identity(d) + self.X_LTS.T.dot(self.X_LTS)
        msxs = np.zeros((d,1))
        for i in range(t):
            msxs = msxs + (self.m_LTS[i]*self.X_LTS[i]).reshape((d,1))
        theta_e =  np.linalg.inv(Vt).dot(msxs)
        eta_t = np.random.multivariate_normal(np.zeros(d), exp_coeff*np.eye(d)).reshape(d,1)
        beta = R*np.sqrt(d*np.log((1 + ((t-1)*L**2)/lam)/delta)) + np.sqrt(lam)*S
        theta_til = theta_e + beta*np.linalg.inv(sqrtm(Vt)).dot(eta_t)
        
        Dst = [np.append(np.fromiter(context.values(), dtype=int), np.array([safe_norm_ins])).reshape(d,1)]

        #  Construct the estimated safe actions set
        for ins in ins_normalized:
            x = np.append(np.fromiter(context.values(), dtype=int), np.array([ins])).reshape(d,1)
            check1 = x.T.dot(theta_e) - beta*np.sqrt(x.T.dot(np.linalg.inv(Vt)).dot(x))
            check2 = x.T.dot(theta_e) + beta*np.sqrt(x.T.dot(np.linalg.inv(Vt)).dot(x))
            if (check1 >= bg_omin and check2 <= bg_omax):
                Dst.append(x)

        Dstresult = [abs(target_bg - Dst[i].T.dot(theta_til)) for i in range(len(Dst))]
        ins_rec = Dst[np.argmin(Dstresult)][d-1][0]

        #print('Dose: {}'.format(ins_rec))
        return ins_rec


    def _acqu_with_lelts(self, context, ins_feasible, S, L, lam, exp_coeff):
        
        #print('+-+- LE-LTS -+-+') 
        target_bg = self.postBG_target_value
        R = 5  # the measurement noise should be noise is R-sub-gaussian for example if the noise is gaussian its variance must be at most R^2
        d = len(self.variables)
        t = 0 if self.X_LTS is None else self.X_LTS.shape[0]  # round (rec.) number
        lam = lam  # regularizaton constant for ridge regression
        delta = 1e-12  # with at least 1 - delta probability.. 
        S = S # maximum value of one element of theta.
        L = L  # max val of context elements
        ins_normalized = ins_feasible
    
        Vt = lam*np.identity(d) + self.X_LTS.T.dot(self.X_LTS)
        msxs = np.zeros((d,1))
        for i in range(t):
            msxs = msxs + (self.m_LTS[i]*self.X_LTS[i]).reshape((d,1))
        theta_e =  np.linalg.inv(Vt).dot(msxs)
        eta_t = np.random.multivariate_normal(np.zeros(d), exp_coeff*np.eye(d)).reshape(d,1)
        beta = R*np.sqrt(d*np.log((1 + ((t-1)*L**2)/lam)/delta)) + np.sqrt(lam)*S
        theta_til = theta_e + beta*np.linalg.inv(sqrtm(Vt)).dot(eta_t)

        #  Choose the best safe action from safe actions set
        Dstresult = [abs(target_bg - np.append(np.fromiter(context.values(), dtype=int), np.array([ins])).reshape(d,1).T.dot(theta_til)) for ins in ins_normalized]
        ins_rec = ins_normalized[np.argmin(Dstresult)]

        #print('Dose: {}'.format(ins_rec))
        return ins_rec


    def update_lts(self, new_x, new_m):
        self.X_LTS = new_x if self.X_LTS is None else np.concatenate((self.X_LTS, new_x), axis=0)
        self.m_LTS = new_m if self.m_LTS is None else np.concatenate((self.m_LTS, new_m), axis=0)



    
