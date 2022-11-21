from scipy.optimize import minimize, Bounds

import numpy as np

def OLR_MTS_SO_pBef_thAf(T,D,B,K,A,Q,P,Theta,m,nu,mu,nuExt,muExt,discountFactor):
#useful functions for period+slots
    def f(x):
        output=0
        for k in range(K):
            output += -a_k[k]*np.log(np.dot(x[:m],theta) + np.dot(x[(k+1)*m:(k+2)*m],theta) +1)
        return output

    def g(x):
        term2=0
        for k in range(1,K+1):
            term2 += np.dot(x[k*m:(k+1)*m],q_k[k-1])
        return np.dot(x[:m],p_t) + term2 - B

    def gradient_f(x):
        output=np.zeros(m*(K+1))
        for k in range(K):
            output[:m] += (-a_k[k]/(1 + np.dot(theta, x[:m]) + np.dot(theta, x[(k+1)*m:(k+2)*m]) ))*theta
        for k in range(1,K+1):
            output[k*m:(k+1)*m] = (-a_k[k-1]/(1 + np.dot(theta, x[:m]) + np.dot(theta, x[k*m:(k+1)*m]) ))*theta
        return output

    def lagrangian(x):
        primal_vars = x
        primal_prev = x_prev
        dual_var = l
        L = np.dot(gradient_f(primal_prev),(primal_vars-primal_prev)) + dual_var*g(primal_vars) + np.linalg.norm(primal_vars-primal_prev)**2/(2*nu)
        return L

    #useful functions for slot updates
    def slot_f(x):
        return -a*np.log(1 + np.dot(x,theta) + np.dot(theta,x_prev[:m]))

    def slot_g(x):
        return np.dot(q,x) - B_slot

    def slot_gradient_f(x):
        return (-a/(1 + np.dot(x,theta) + np.dot(theta,x_prev[:m])))*theta

    def slot_lagrangian(x):
        primal_vars = x
        primal_prev = y_prev
        dual_var = l_slot
        L = np.dot(slot_gradient_f(primal_prev),(primal_vars-primal_prev)) + dual_var*slot_g(primal_vars) + np.linalg.norm(primal_vars-primal_prev)**2/(2*nuExt)
        return L

    #prepare traces
    A_k_it = A[:K*T+K]
    A_k_it = A_k_it.reshape(T+1,K)
    
    Q_k_it = Q[:K*T+K,:]
    
    P = P*(K/discountFactor)

    t=1
    
    #initialize output arrays
    revenue=0
    results=np.array([])
    fit=0
    vect_fit=np.array([])
    
    ext_revenue=0
 
    ext_results=np.array([])
 
    ext_fit=0
   
    ext_vect_fit=np.array([])
    
    #intialize primal and dual
    x_prev = np.zeros(m*(K+1))
    l=0
    y_prev = np.zeros(m)
    l_slot=0

    Xprev=np.empty((T,m*(K+1)))
    while t <=T:
        #previous demand, price and utility
        a_k = A_k_it[t-1]
        p_t = P[t]          # it depends if p_t is given or not by the NO
        q_k = Q_k_it[(t-1)*K:t*K,:]
        theta = Theta[t-1]    #utility of previous period

        #bounds
        my_lb = [0]*(m*(K+1))
        my_ub = D*(K+1)
        my_bounds=Bounds(my_lb,my_ub)

        #solve (19)
        res = minimize(lagrangian, np.zeros((K+1)*m), bounds=my_bounds)
        x_prev=res.x
        Xprev[t-1]=x_prev
        
        B_slot = (B - np.dot(x_prev[:m],p_t))/K
        
        k=0
        slots=np.array([])
        while k<K:
            #price and demand q and a
            if k==0:
                q = Q_k_it[t*K+k-1]
                a = A_k_it[t-1][K-1]
            else:
                q = Q_k_it[t*K+k-1]
                a = A_k_it[t][k-1]
            
            #solve per slot lagrangian
            lb=np.zeros(m)
            ub=np.array(D)
            bounds=Bounds(lb,ub)
            Res=minimize(slot_lagrangian, np.zeros(m), bounds=bounds)
            y_prev=Res.x
            
            slots=np.append(slots, y_prev)
            
            #update q to compute the slot dual
            q=Q_k_it[t*K+k]
            
            l_slot = max(0, l_slot + muExt*slot_g(y_prev))
            
            k=k+1
            

        #update price and demand and theta
        a_k = A_k_it[t]
        #p_t = P[t]
        q_k = Q_k_it[t*K:(t+1)*K,:]
        theta = Theta[t]
        
        
        #calculate regret and fit
        revenue += f(x_prev)
        results = np.append(results, revenue/t)

        fit += g(x_prev)
        vect_fit = np.append(vect_fit, fit/t)
        
        #update dual
        l=max(0,l+mu*g(x_prev))
        #l=max(0,l + np.dot(p_t,x_prev[:m]) + np.dot(q_k,slots) - B )
        
        #extension OLR_MTS_SO
        ext_revenue += f( np.concatenate((x_prev[:m],slots)) )
        
        ext_results=np.append(ext_results, ext_revenue/t)
        
        q_k = q_k.reshape(m*K,)
        ext_fit += (np.dot(x_prev[:m],p_t) + np.dot(q_k,slots) - B)
        
        ext_vect_fit=np.append(ext_vect_fit, ext_fit/t)


        t=t+1
        
    perf = -revenue
    ext_perf = -ext_revenue
    
    return perf, ext_perf, results, vect_fit, ext_results, ext_vect_fit, Xprev
