import numpy as np
import math

def benchmark(T,D,B,K,A,Q,P,discountFactor):
    
    alpha=5/3
    a=2
    p=q=2
    nu=0.1
    mu=0.1
    gap1,gap2,gap3=1,0.5,0.1
    
    
    
    #constant of the problem
    E = D*math.sqrt(K+1)
    G = a*math.sqrt(K*(K+1))
    M = max(D*(p+K*q)-B,B)
    
    import ipopt
    
    class probleme(object):
        def __init__(self):
            pass

        def objective(self, x):
            output=0
            for k in range(K):
                output += -a_k[k]*np.log(x[0]+x[k+1]+1)
            return output
            #The callback for calculating the objective


        def gradient(self, x):
            # The callback for calculating the gradient
            output=np.zeros(K+1)
            for k in range(K):
                output[0] += -a_k[k]/(1+x[0]+x[k+1])
            for k in range(1,K+1):
                output[k] = -a_k[k-1]/(1+x[0]+x[k])
            return output



        def constraints(self, x):
            #
            # The callback for calculating the constraints
            #
            return np.dot(mult_t,x)

        def jacobian(self, x):
            #
            # The callback for calculating the Jacobian
            #
            return mult_t
    
    A_k_it = A[:K*T+K]
    A_k_it = A_k_it.reshape(T+1,K)
    
    Q_k_it = Q[:K*T+K]
    Q_k_it = Q_k_it.reshape(T+1,K)
    
    new_column = P*(K/discountFactor)
    Mult_t = np.insert(Q_k_it, 0, new_column, axis=1)
    
    
    x0 = np.ones(K+1).tolist()

    lb = np.zeros(K+1).tolist()
    ub = (np.ones(K+1)*D).tolist()

    cl = [0.0]
    cu = [B]


    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=probleme(),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )
    
    t=0
    opt_revenue=0
    opt_results=np.array([])
    opt_fit=0
    opt_vect_fit=np.array([])
    Xopt=np.empty((T+1,K+1))
    U_z_T = 0
    U_g_T = 0
    U_z = []
    U_g = []
    Bth1 = []
    Bth2 = []
    Bth3 = []
    VTH1 = []
    VTH2 = []
    VTH3 = []
    while t <=T:
        #udpdate demand and price
        a_k = A_k_it[t]
        mult_t = Mult_t[t]
        #benchmark
        x, info = nlp.solve(x0)
        Xopt[t]=x
        
        if t>=1:
            #compute the theoretical bound Bth
            
            #U_z
            U_z_T += np.linalg.norm(Xopt[t-1]-Xopt[t])
            U_z.append(U_z_T/t**alpha)
            
            #U_g
        
            if P[t] - P[t-1] > 0:
                res = D
            else:
                res = 0
            res1 = (P[t]-P[t-1])*res
            
            resk = 0
            for k in range(0,K,1):
                if Q[(t-1)*K+k] > Q[t*K+k]:
                    res=0
                else:
                    res=D
                resk += (Q[t*K+k]-Q[(t-1)*K+k])*res
                
            U_g_T += (res1 +resk)
            U_g.append(U_g_T/t**alpha)
            
            Bth_over_t1 = E*U_z[-1]/nu + nu*G**2/(2*t**(alpha-1)) + (t+1)*mu*M**2/(2*t**alpha) + E**2/(2*nu*t**alpha) + U_g[-1]*mu*M + U_g[-1]*2*E*G/gap1 + U_g[-1]*E**2/(2*nu*gap1) + U_g[-1]*mu*M**2/(2*gap1)
            
            Bth_over_t2 = E*U_z[-1]/nu + nu*G**2/(2*t**(alpha-1)) + (t+1)*mu*M**2/(2*t**alpha) + E**2/(2*nu*t**alpha) + U_g[-1]*mu*M + U_g[-1]*2*E*G/gap2 + U_g[-1]*E**2/(2*nu*gap2) + U_g[-1]*mu*M**2/(2*gap2)
            
            Bth_over_t3 = E*U_z[-1]/nu + nu*G**2/(2*t**(alpha-1)) + (t+1)*mu*M**2/(2*t**alpha) + E**2/(2*nu*t**alpha) + U_g[-1]*mu*M + U_g[-1]*2*E*G/gap3 + U_g[-1]*E**2/(2*nu*gap3) + U_g[-1]*mu*M**2/(2*gap3)
            
            Vth1 = M/t + 2*E*G/(mu*gap1*t) + E**2/(2*nu*mu*gap1*t) + M**2/(2*gap1*t)
            
            Vth2 = M/t + 2*E*G/(mu*gap2*t) + E**2/(2*nu*mu*gap2*t) + M**2/(2*gap2*t)
            
            Vth3 = M/t + 2*E*G/(mu*gap3*t) + E**2/(2*nu*mu*gap3*t) + M**2/(2*gap3*t)
            
            
            
            VTH1.append(Vth1)
            VTH2.append(Vth2)
            VTH3.append(Vth3)
            
            Bth1.append(Bth_over_t1)
            Bth2.append(Bth_over_t2)
            Bth3.append(Bth_over_t3)
                
            
  
   
            opt_revenue += info['obj_val']
            opt_results=np.append(opt_results,opt_revenue/t)
        
        

            opt_fit += (info['g']-B)
            opt_vect_fit=np.append(opt_vect_fit, opt_fit/t)
        
        t=t+1
        
    opt_perf = -opt_revenue
        
    return opt_perf, opt_results, opt_vect_fit, U_z_T, U_g, Bth1, Bth2, Bth3, VTH1, VTH2, VTH3