import numpy as np

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

# state transistion function
def f(t, x):
  global Vb, Vdr, Vqr, Pt
  Iqs, Ids, Eq, Ed, wr, wt, theta_r, theta_t = x.T
  Ia = Iqs + 1j*Ids

  Vds = np.imag(Vb+Ia*1j*Xe)
  Vqs = np.real(Vb+Ia*1j*Xe)

  theta_d = theta_t - theta_r

  dtheta_t = wb*(wt - ws)
  dtheta_r = wb*(wr - ws)
  dtheta_d = dtheta_t - dtheta_r


  a = theta_d + (cd/ks) * dtheta_d - delta
  b = theta_d + (cd/ks) * dtheta_d + delta

  a = np.heaviside(a, 0)
  b = 1 - np.heaviside(b, 0)

  Tsa = ks*(theta_d - delta) + cd*dtheta_d
  Tsb = ks*(theta_d + delta) + cd*dtheta_d

  Ts = a*(1-b)*Tsa + (1-a)*b*Tsb

  Tt = Pt/wt
  Te = (Eq*Iqs + Ed*Ids)/ws

  dwt = (Tt - Ts)/(2*Ht)
  dwr = (Ts - Te)/(2*Hg)

  dIqs = (-R1*Iqs + Ls_prime*Ids + wr*Eq - Ed/Tr - Vqs + km*Vqr)/k1
  dIds = (-R1*Ids - Ls_prime*Iqs + wr*Ed + Eq/Tr - Vds + km*Vdr)/k1

  dEq = (R2*Ids - Eq/Tr + (1-wr)*Ed - km*Vdr)*wb
  dEd = (-R2*Iqs - Ed/Tr - (1-wr)*Eq + km*Vqr)*wb

  dX = np.array([dIqs, dIds, dEq, dEd, dwr, dwt, dtheta_r, dtheta_t]).T
  return dX

# measurement function
def h(x):
  global Vb, Vdr, Vqr, Pt
  Iqs, Ids, Eq, Ed, wr, wt, theta_r, theta_t = x.T

  Ia = Iqs + 1j*Ids

  Vds = np.imag(Vb+Ia*1j*Xe)
  Vqs = np.real(Vb+Ia*1j*Xe)

  Iqr = -Ed/(ws*Lm) - km*Iqs
  Idr = -Eq/(ws*Lm) - km*Ids
  Pe = Vds*Ids + Vqs*Iqs + Vdr*Idr + Vqr*Iqr
  Qe = Vds*Iqs + Vqs*Ids + Vdr*Iqr + Vqr*Idr

  if(x.ndim > 1):
    Y = np.zeros((x.shape[0], 3))
    Y[:, 0] = wr
    Y[:, 1] = Pe
    Y[:, 2] = Qe
  else:
    Y = np.zeros(3)
    Y[0] = wr
    Y[1] = Pe
    Y[2] = Qe

  return Y