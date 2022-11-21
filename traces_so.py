import numpy as np
import math

def traces_so(a,q,p,theta,T,m,boolean, sinPeriod, sinAmp):
    K_range=np.arange(3,31,1)
    Kmax=K_range[-1]

    A_k=np.array([])
    Q_k=np.zeros((1,m))
    P_t=np.zeros((1,m))
    Theta_t=np.zeros((1,m))

    if boolean==0:
        for k in range(Kmax+Kmax*T):

            a_k = np.random.uniform(0,a)

            q_k = np.random.uniform(0,q,m)
            
            A_k = np.append(A_k, a_k)

            Q_k = np.vstack((Q_k, q_k))

        for t in range(T+1):

            p_t = np.random.uniform(0,p,m)

            theta_t = np.random.uniform(0,theta,m)

            P_t = np.vstack((P_t, p_t))

            Theta_t = np.vstack((Theta_t, theta_t))
            
    else:
        for k in range(Kmax+Kmax*T):

            q_k = sinAmp*np.sin(k*math.pi/sinPeriod) + np.random.uniform(sinAmp,q,m)

            Q_k = np.vstack((Q_k, q_k))

            a_k = sinAmp*np.sin(k*math.pi/sinPeriod) + np.random.uniform(sinAmp,a)

            A_k = np.append(A_k, a_k)

        for t in range(T+1):

            p_t = sinAmp*np.sin(math.pi*t/sinPeriod) + np.random.uniform(sinAmp,p,m)

            P_t = np.vstack((P_t, p_t))

            theta_t = sinAmp*np.sin(math.pi*t/sinPeriod) + np.random.uniform(sinAmp,theta,m)

            Theta_t = np.vstack((Theta_t, theta_t))
            
    #delete the first row full of zeros that we need to use vstack, function of numpy
    Q_k = np.delete(Q_k, 0, 0)
    P_t = np.delete(P_t, 0, 0)
    Theta_t = np.delete(Theta_t, 0, 0)

    return A_k, Q_k, P_t, Theta_t
