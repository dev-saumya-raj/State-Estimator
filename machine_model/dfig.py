import numpy as np

class DFIG:

    Xe = 0.0006
    Xm = 6
    Rs = Xm/800
    Rr = 1.1*Rs
    Lm = Xm
    Ls = 1.01*Lm
    Lr = 1.005*Ls
    ks = 0.3
    cd = 0.01
    Ht = 2
    Hg = 0.1*Ht
    ws = 1
    wb = 314
    delta = 0.1
    Ls_prime = Ls-(Lm**2)/Lr
    k1 = Ls_prime/wb
    Tr = Lr/Rr
    km = Lm/Lr
    R2 = (km**2)*Rr
    R1 = R2 + Rs

    k1 = Ls_prime/wb
    Tr = Lr/Rr
    km = Lm/Lr
    R2 = (km**2)*Rr
    R1 = R2 + Rs

    def __init__(self):
        self.inputs = np.zeros((4,)) # Vb, Vdr, Vqr, Pt
    
    def f(self, t, x):
        Vb, Vdr, Vqr, Pt = self.inputs
        Iqs, Ids, Eq, Ed, wr, wt, theta_r, theta_t = x
        Ia = Iqs + 1j*Ids

        Vds = np.imag(Vb+Ia*1j*self.Xe)
        Vqs = np.real(Vb+Ia*1j*self.Xe)

        theta_d = theta_t - theta_r

        dtheta_t = self.wb*(wt - self.ws)
        dtheta_r = self.wb*(wr - self.ws)
        dtheta_d = dtheta_t - dtheta_r


        a = theta_d + (self.cd/self.ks) * dtheta_d - self.delta
        b = theta_d + (self.cd/self.ks) * dtheta_d + self.delta

        a = np.heaviside(a, 0)
        b = 1 - np.heaviside(b, 0)

        Tsa = self.ks*(theta_d - self.delta) + self.cd*dtheta_d
        Tsb = self.ks*(theta_d + self.delta) + self.cd*dtheta_d

        Ts = a*(1-b)*Tsa + (1-a)*b*Tsb

        Tt = Pt/wt
        Te = (Eq*Iqs + Ed*Ids)/self.ws

        dwt = (Tt - Ts)/(2*self.Ht)
        dwr = (Ts - Te)/(2*self.Hg)

        dIqs = (-self.R1*Iqs + self.Ls_prime*Ids + wr*Eq - Ed/self.Tr - Vqs + self.km*Vqr)/self.k1
        dIds = (-self.R1*Ids - self.Ls_prime*Iqs + wr*Ed + Eq/self.Tr - Vds + self.km*Vdr)/self.k1

        dEq = ( self.R2*Ids - Eq/self.Tr + (1-wr)*Ed - self.km*Vdr)*self.wb
        dEd = (-self.R2*Iqs - Ed/self.Tr - (1-wr)*Eq + self.km*Vqr)*self.wb

        dX = np.array([dIqs, dIds, dEq, dEd, dwr, dwt, dtheta_r, dtheta_t])
        return dX
    
    def h(self, x):
        Vb, Vdr, Vqr, Pt = self.inputs
        Iqs, Ids, Eq, Ed, wr, wt, theta_r, theta_t = x

        Ia = Iqs + 1j*Ids

        Vds = np.imag(Vb+Ia*1j*self.Xe)
        Vqs = np.real(Vb+Ia*1j*self.Xe)

        Iqr = -Ed/(self.ws*self.Lm) - self.km*Iqs
        Idr = -Eq/(self.ws*self.Lm) - self.km*Ids
        Pe = Vds*Ids + Vqs*Iqs + Vdr*Idr + Vqr*Iqr
        Qe = Vds*Iqs + Vqs*Ids + Vdr*Iqr + Vqr*Idr

        Y = np.zeros(3)
        Y[0] = wr
        Y[1] = Pe
        Y[2] = Qe

        return Y