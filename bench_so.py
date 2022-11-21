import numpy as np
import math
#x decision variable must be a 1-D array, for ipopt solver.

def benchmark_so(T,D,B,K,A,Q,P,Theta,m,discountFactor):  #K must be less than Kmax, D \in R^m
    
    alpha=5/3
    a=2
    p=q=2
    theta=2
    nu=0.1
    mu=0.1
    gap1,gap2,gap3=1,0.5,0.1
    
    #constant of the problem
    sum_carres = 0
    for d in D:
        sum_carres += d**2
    F = math.sqrt(sum_carres)*math.sqrt(K+1)
    G = a*theta*math.sqrt(m*K*(K+1))
    M = max(sum(D)*(p+K*q)-B,B)
    
    import ipopt
    
    class probleme(object):
        def __init__(self):
            pass

        def objective(self, x):
            output=0
            for k in range(K):
                output += -a_k[k]*np.log(1+np.dot(theta,x[:m])+np.dot(theta,x[(k+1)*m:(k+2)*m]))
            return output
            #The callback for calculating the objective


        def gradient(self, x):
            # The callback for calculating the gradient
            output=np.zeros((K+1)*m)
            for k in range(K):
                output[:m] += -a_k[k]/(1+x[:m]+x[(k+1)*m:(k+2)*m])
            for k in range(1,K+1):
                output[k*m:(k+1)*m] = -a_k[k-1]/(1+x[:m]+x[k*m:(k+1)*m])
            return output



        def constraints(self, x):
            #
            # The callback for calculating the constraints
            #
            term2 = 0
            for k in range(K):
                term2 += np.dot(x[(k+1)*m:(k+2)*m],q_k[k])
            return np.dot(x[:m],p_t) + term2

        def jacobian(self, x):
            #
            # The callback for calculating the Jacobian
            #
            return np.insert(q_k, 0, p_t, axis=0) #insert p_t as first row
    
    A_k_it = A[:K*T+K]
    A_k_it = A_k_it.reshape(T+1,K)
    
    Q_k_it = Q[:K*T+K,:]
    #Q_k_it = Q_k_it.reshape(T+1,K)
    
    P=P*(K/discountFactor)
    #new_column = P
    #Mult_t = np.insert(Q_k_it, 0, new_column, axis=1)   #numpy insert function example-how to add new column at position 0
    
    
    x0 = np.ones((K+1)*m).tolist()

    lb = np.zeros((K+1)*m).tolist()
    #ub = np.array([D,]*(K+1)).tolist()   #this is for x as 2-D array
    ub = D*(K+1)
    

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
    Xopt=np.empty((T+1,m*(K+1)))
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
        #print(a_k.shape)
        
        theta = Theta[t]
        #print(theta.shape)

        p_t = P[t]

        q_k = Q_k_it[t*K:(t+1)*K,:]
        
        #benchmark
        x, info = nlp.solve(x0)
        Xopt[t]=x
        
        if t>=1:
                      
            
            #compute the theoretical bound Bth
            
            #U_z
            U_z_T += np.linalg.norm(Xopt[t-1]-Xopt[t])
            U_z.append(U_z_T/t**alpha)
            
            #U_g
            res1 = 0
            for l in range(m):
                if P[t][l] - P[t-1][l] > 0:
                    res = D[l]
                else:
                    res = 0
                res1 += (P[t][l]-P[t-1][l])*res
            
            resk = 0
            for l in range(m):
                for k in range(0,K,1):
                    if Q[(t-1)*K+k][l] > Q[t*K+k][l]:
                        res=0
                    else:
                        res=D[l]
                    resk += (Q[t*K+k][l]-Q[(t-1)*K+k][l])*res
                
            U_g_T += (res1 +resk)
            U_g.append(U_g_T/t**alpha)
            
            Bth_over_t1 = F*U_z[-1]/nu + nu*G**2/(2*t**(alpha-1)) + (t+1)*mu*M**2/(2*t**alpha) + F**2/(2*nu*t**alpha) + U_g[-1]*mu*M + U_g[-1]*2*F*G/gap1 + U_g[-1]*F**2/(2*nu*gap1) + U_g[-1]*mu*M**2/(2*gap1)
            
            Bth_over_t2 = F*U_z[-1]/nu + nu*G**2/(2*t**(alpha-1)) + (t+1)*mu*M**2/(2*t**alpha) + F**2/(2*nu*t**alpha) + U_g[-1]*mu*M + U_g[-1]*2*F*G/gap2 + U_g[-1]*F**2/(2*nu*gap2) + U_g[-1]*mu*M**2/(2*gap2)
            
            Bth_over_t3 = F*U_z[-1]/nu + nu*G**2/(2*t**(alpha-1)) + (t+1)*mu*M**2/(2*t**alpha) + F**2/(2*nu*t**alpha) + U_g[-1]*mu*M + U_g[-1]*2*F*G/gap3 + U_g[-1]*F**2/(2*nu*gap3) + U_g[-1]*mu*M**2/(2*gap3)
            
            Vth1 = M/t + 2*F*G/(mu*gap1*t) + F**2/(2*nu*mu*gap1*t) + M**2/(2*gap1*t)
            
            Vth2 = M/t + 2*F*G/(mu*gap2*t) + F**2/(2*nu*mu*gap2*t) + M**2/(2*gap2*t)
            
            Vth3 = M/t + 2*F*G/(mu*gap3*t) + F**2/(2*nu*mu*gap3*t) + M**2/(2*gap3*t)
            
            
           
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
        
    opt_perf = - opt_revenue
        
    return opt_perf, opt_results, opt_vect_fit, U_g, U_z, Bth1, Bth2, Bth3, VTH1, VTH2, VTH3