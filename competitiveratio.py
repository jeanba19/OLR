import numpy as np

def comp(T,D,B,K,A,Q,P,discountFactor):
    
    import ipopt
    
    class probleme(object):
        def __init__(self):
            pass

        def objective(self, x):
            output=0
            for t in range(T+1):
                for k in range(1,K+1):
                    output += -A_k_it[t][k-1]*np.log(x[t*(K+1)]+x[k+t*(K+1)]+1)
            return output
                #The callback for calculating the objective


        def gradient(self, x):
            # The callback for calculating the gradient
            output=np.zeros((T+1)*(K+1))
            for t in range(T+1):
                for k in range(1,K+1):
                    output[t*(K+1)] += -A_k_it[t][k-1]/(1 + x[t*(K+1)] + x[t*(K+1)+k] )
                    
                    output[t*(K+1)+k] = -A_k_it[t][k-1]/(1 + x[t*(K+1)] + x[t*(K+1)+k] )
                    
            return output
           



        def constraints(self, x):
            #
            # The callback for calculating the constraints
            #
            output = 0
            for t in range(T+1):
                output += P[t]*x[t*(K+1)]
                for k in range(1,K+1):
                    output += Q_k_it[t][k-1]*x[t*(K+1)+k]
            return output

        def jacobian(self, x):
            #
            # The callback for calculating the Jacobian
            #
            output = np.zeros((T+1)*(K+1))
            for t in range(T+1):
                output[t*(K+1)] = P[t]
                for k in range(1,K+1):
                    output[t*(K+1)+k] = Q_k_it[t][k-1]
            return output
    
    A_k_it = A[:K*T+K]
    A_k_it = A_k_it.reshape(T+1,K)
    
    Q_k_it = Q[:K*T+K]
    Q_k_it = Q_k_it.reshape(T+1,K)
    
    P = P*(K/discountFactor)
    
    #Mult_t = np.insert(Q_k_it, 0, new_column, axis=1)
    
    
    x0 = np.ones((T+1)*(K+1)).tolist()

    lb = np.zeros((T+1)*(K+1)).tolist()
    ub = (np.ones((T+1)*(K+1))*D).tolist()

    cl = [0.0]
    cu = [B*(T+1)]


    nlp = ipopt.problem(
                n=len(x0),
                m=len(cl),
                problem_obj=probleme(),
                lb=lb,
                ub=ub,
                cl=cl,
                cu=cu
                )
    
    #solve the major problem \mathbb P
    
    x, info = nlp.solve(x0)
    
    #once the whole sequence x is obtained, we derive the regret and fit using a loop over T
    
    t=1
    opt_revenue=0
    opt_results=np.array([])
    opt_fit=0
    opt_vect_fit=np.array([])
    
    
    Xopt = x.reshape((T+1,K+1))
    
    
    U_z_T = 0
    
    while t <=T:
        
        
     
            
        U_z_T += np.linalg.norm(Xopt[t-1]-Xopt[t])
        
        for k in range(1,K+1):
                    opt_revenue += -A_k_it[t][k-1]*np.log(x[t*(K+1)]+x[k+t*(K+1)]+1)

        
        opt_results=np.append(opt_results,opt_revenue/t)



        opt_fit += P[t]*x[t*(K+1)]
        
        for k in range(1,K+1):
            opt_fit += Q_k_it[t][k-1]*x[t*(K+1)+k]
            
        opt_fit = opt_fit - B
        
        opt_vect_fit=np.append(opt_vect_fit, opt_fit/t)
        
        t=t+1
        
        
    return opt_results, opt_vect_fit, U_z_T, Xopt