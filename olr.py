import numpy as np
from scipy.optimize import minimize, Bounds

def OLR(T,D,B,K,A,Q,P,nu,mu,nuExt,muExt, discountFactor):
    
    #useful functions for period+slots
    def f(x):
        output=0
        for k in range(K):
            output += -a_k[k]*np.log(x[0]+x[k+1]+1)
        return output

    def g(x):
        return np.dot(mult_t,x) - B 

    def gradient_f(x):
        output=np.zeros(K+1)
        for k in range(K):
            output[0] += -a_k[k]/(x[0]+x[k+1]+1)
        for k in range(1,K+1):
            output[k] = -a_k[k-1]/(x[0]+x[k]+1)
        return output

    def lagrangian(x):
        primal_vars = x
        primal_prev = x_prev
        dual_var = l
        L = np.dot(gradient_f(primal_prev),(primal_vars-primal_prev)) + dual_var*g(primal_vars) + np.linalg.norm(primal_vars-primal_prev)**2/(2*nu)
        return L

    #useful functions for slot updates  not useful commented ##
    def slot_f(x):
        return -a*np.log(1 + x + x_prev[0])

    def slot_g(x):
        return q*x - B_slot

    def slot_gradient_f(x):
        return -a/(1 + x + x_prev[0])

    def slot_lagrangian(x):
        primal_vars = x
        primal_prev = y_prev
        dual_var = l_slot
        L = slot_gradient_f(primal_prev)*(primal_vars-primal_prev) + dual_var*slot_g(primal_vars) + (primal_vars-primal_prev)**2/(2*nuExt)
        return L
    

    
    
    #prepare traces
    A_k_it = A[:K*T+K]
    A_k_it = A_k_it.reshape(T+1,K)

    Q_k_it = Q[:K*T+K]
    Q_k_it = Q_k_it.reshape(T+1,K)

    #new_column = P*(K/discountFactor)
    P = P*(K/discountFactor)
    #Mult_t = np.insert(Q_k_it, 0, new_column, axis=1)
    
    
    
    t=1
    #initialize primal and dual
    x_prev=np.zeros(K+1)
    y_prev=0

    l=0
    l_slot=0

    
    #initialize output arrays
    revenue=0
    results=np.array([])
    fit=0
    vect_fit=np.array([])

    ext_revenue=0

    ext_results=np.array([])
 
    ext_fit=0

    ext_vect_fit=np.array([])

    X_prev = np.empty((T,K+1))
    X_mts = np.empty((T,K+1))
    
    while t <=T:
        #set price q_k and demand a_k to the values of the previous period
        q_k = Q_k_it[t-1]
        a_k = A_k_it[t-1]

        #update price p_t before decision/or after decision
        #p_t = np.array([P[t]])
        p_t = np.array([P[t-1]])
        mult_t = np.concatenate((p_t,q_k))

        #solve (12)
        my_lb=np.zeros(K+1)
        my_ub=np.ones(K+1)*D
        my_bounds=Bounds(my_lb,my_ub)
        res=minimize(lagrangian, np.zeros(K+1), bounds=my_bounds)
        x_prev=res.x
        
        X_prev[t-1]=x_prev
        
        #OLR-MTS slots updates
        
        B_slot = (B - x_prev[0]*p_t)/K

        k=0
        slots=np.array([])
        while k<K:
            #price and demand q and a
            if k == 0:
                q = Q_k_it[t-1][K-1]
                a = A_k_it[t-1][K-1]
            else:
                q = Q_k_it[t][k-1]
                a = A_k_it[t][k-1]
            #solve (12)
            lb=np.array([0])
            ub=np.array([D])
            bounds=Bounds(lb,ub)
            rEs=minimize(slot_lagrangian, [0], bounds=bounds)
            
            y_prev=rEs.x

            slots=np.append(slots, y_prev)

            #update q to compute the dual
            q=Q_k_it[t][k]
            
            l_slot = max(0,l_slot+muExt*slot_g(y_prev))

            k=k+1

        
        #udpdate demand and price
        a_k = A_k_it[t]
        q_k = Q_k_it[t]
        p_t = np.array([P[t]])
        mult_t = np.concatenate((p_t,q_k))

        #extensions OLR_MTS
        X_mts[t-1] = np.concatenate((np.array([x_prev[0]]),slots))
        
        ext_revenue += f(np.concatenate((np.array([x_prev[0]]),slots)))

        
        ext_results=np.append(ext_results, ext_revenue/t)
 

        ext_fit += (p_t*x_prev[0] + np.dot(q_k,slots) - B)
 
        
        ext_vect_fit=np.append(ext_vect_fit, ext_fit/t)
      



        #OLR average cost
        revenue += f(x_prev)
        results=np.append(results, revenue/t)

        fit += g(x_prev)
        vect_fit=np.append(vect_fit,fit/t)


        #update dual
        l=max(0,l+mu*g(x_prev))


        t=t+1
    
    return results, vect_fit, ext_results, ext_vect_fit, X_prev, X_mts
