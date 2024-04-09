# IMPORTS ---

from functools import partial
from copy import copy
from numpy import pi, arange, arctan, dot, exp, flip, sin, cos, sqrt, linspace, outer, ones, zeros, zeros_like, full, full_like
from numpy.polynomial.legendre import leggauss
from numpy.linalg import inv
from matplotlib.pyplot import plot, xlim, ylim, xlabel, ylabel, grid, legend, title, yscale, show, figure
from matplotlib.animation import FuncAnimation

# RECURRING FUNCTIONS ---
# [TODO]: use adequate func names
def chebyshev(N,Noff,M,r,L0,beta=False,base=None):
    y = zeros([N+Noff,M])
    dy = copy(y)
    ddy = copy(y)
    z = arctan(L0/r)

    f = lambda i,j=1: 2*i+j
    for i in range(N+Noff):
      (a,b) = (f(i),f(i)) if beta else (f(i+0.5),f(i,2))
      s = sin(a*z)
      d = cos(b*z)*L0
      c = -sin(b*z)*b**2
      e = 1+L0**2/r**2
      y[i,] = s
      dy[i,] = -d/(r**2*e)
      ddy[i,] = c*L0**2/(r**4*e**2)+2*d/(r**3*e)-2*d**3/(r**5*e**2)
    (py, pdy, pddy) = (y[0:base,:], dy[0:base,:], ddy[0:base,:]) if base else (_, _, _)
    return (y, dy, ddy, py, pdy, pddy)

def alphaphi(ax,s,ra,rb=0):
  s = s**2
  return ax*(exp(-(ra-rb)**2/s)+exp(-(ra+rb)**2/s))

def plot_data(r, a, b=None, color='--r', xlim_a=None, xlim_b=None,
        xlb=None, ylb=None, ttl=None, lbl=None,
        has_legend=False, log_y=False):
    if b is not None: plot(r, a, r, b, color, label=lbl)
    else: plot(r, a, color, label=lbl)
    if (xlim_a and xlim_b): xlim(xlim_a,xlim_b)
    if xlb: xlabel(xlb)
    if ylb: ylabel(ylb)
    if has_legend: legend()
    if ttl: title(ttl)
    if log_y: yscale("log")
    grid()
    show()

# PARAMETERS ---

M = 3000 # plot truncation
N = 200  # Truncation order
L0 = 2   # Map parameter

# Initial conditions of Phi (Scalar field)
r0 = 2
A0 = 0.1
sigma = 1

# Newton-Raphson loop
tol = 1e-18 # tolerance
iterations = 50

# Runge-Kutta 4th order
h = 0.0002 # step size
t = 0
tf = 7.0
epsilon0 = 1
eta0 = 1

# ---

cr = cos(arange(2*N+4)*pi/(2*N+3))[1:N+2] # collocation points
r = flip(L0*cr/(sqrt(1-cr**2))) # physical domain

# Base Matrix (Chebyshev Polinomials):
SB, rSB, _, psi, rpsi, rrpsi = chebyshev(N,3,N+1,r,L0,base=N+1)

inv_psi = inv(psi)
a0 = dot(alphaphi(A0*r**2,sigma,r,r0), inv_psi) # coeficients a(0)
Phi = dot(a0, psi) # approximate solution in t = 0
rPhi= dot(a0, rpsi)

# Initial conditions for Phi
rx = linspace(0.000001,10,M)
_, _, _, psiplot, _, _ = chebyshev(N,1,M,rx,L0,base=N+1)
Phi_x0 = alphaphi(A0*rx**2,sigma,rx,r0)
Phi_x = dot(a0, psiplot)

plot_data(rx,Phi_x,Phi_x0,xlb='r')
plot_data(rx,Phi_x0-Phi_x,xlim_a=0,xlim_b=8,xlb='r',ylb="$|\\phi_N - \\phi_0|$")

# Initial conditions for Alpha:
al0 = dot(alphaphi(1-A0,sigma,r,r0)-1, inv_psi)

# Initial values of Krr and K:
# Base functions:

SB1 = 1/2*(SB[1:(N+2),:] + psi)
rSB1 = 1/2*(rSB[1:(N+2),:] + rpsi)

inv_SB1 = inv(SB1)
K0 = alphaphi(A0/20*r**2,sigma,r)
fk0 = dot(K0, inv_SB1)
K = dot(fk0, SB1)

b0 = dot(zeros(N+1), psi)
Pi = dot(b0, psi)
c0 = full(N+1, 0.001)

rPhi2 = rPhi**2
Pi2 = Pi**2

for _ in range(iterations):
  Chi = dot(c0, psi)
  rChi = dot(c0, rpsi)
  rrChi = dot(c0, rrpsi)

  vx = exp(4*Chi)
  vz = exp(-4*Chi)
  vw = vx*(Pi2 + vz*rPhi2)

  J1 = rChi*rpsi
  H0 = 4*rChi**2 + 4*rrChi + 8/r*rChi - vx*K**2/4 + 1/2*vw
  
  # H0 x inverted JH
  c = c0 - dot(H0, inv(8*J1 + 4*rrpsi + 8/r*rpsi + 4*psi*vx*K/4 - 2*psi*vw + 1/2*vx*(4*psi*vz*rPhi2)))
  if max(abs(c0 - c)) <= tol: break
  c0 = c

plot_data(rx, dot(c0,psiplot), xlim_a=0, xlim_b=8)

# Field Equations
# Base functions for Beta
SB2, rSB2, rrSB2, _, _, _ = chebyshev(N,1,N+1,r,L0,beta=True,base=N+1)

# Quadrature Integration
# Legendre quadrature points
Nq = int(3/2*N) # Quadrature truncation
lg = leggauss(Nq+1)[0]

# Legendre Polinomials
P = zeros([Nq+3,Nq+1])
P[0,] = 1
P[1,] = lg
colP = copy(P)
colP[1,] = 1

for i in range(2,Nq+3):
  P[i,] = ((2*i-1)*lg*P[i-1,] - (i-1)*P[i-2,])/(i)
  colP[i,] = i*P[i-1] + lg*colP[i-1]

wq_col = 2/((1-lg**2)*colP[Nq+1]**2) # Legendre weight
rq = L0*(1+lg)/(1-lg) # Physical quadrature domain
qSB, qrSB, qrrSB, qpsi, rqpsi, rrqpsi = chebyshev(Nq,3,Nq+1,L0,rq,base=N+1)

# Initial Phi in quadrature points
#qPhi = dot(a0, qpsi)
#rqPhi= dot(a0, rqpsi)

# Initial Pi for quadrature points
#qPi = dot(b0, qpsi)

# Initial Chi for quadrature points:
#qChi = dot(c0, qpsi)   # Verificado todos
#rqChi = dot(c0, rqpsi)
#rrqChi = dot(c0, rrqpsi)

qSB1 = 1/2*(qSB[1:(N+2),:] + qpsi)
rqSB1 = 1/2*(qrSB[1:(N+2),:] + rqpsi)
rrqSB1 = 1/2*(qrrSB[1:(N+2),:] + rrqpsi)
#qKrr = dot(ck0, qSB1)

# Alpha na origem
psi_0 = zeros(N+1) # psi(t,0)
for i in range(N+1):
  psi_0[i,] = sin((2*i+1)*pi/2) # arccot(0) = Pi/2

# Filtering
#Nc = 0
#Nf = N - Nc
#coef_f = 36
#s = 10
#filter = exp(- coef_f*((arange(N - Nc + 1))/(N-Nc))**s)
filter = ones(N+1)

M0 = 2*dot(arange(1,2*N+2,2),c0) # Madm(t = 0)

Madm_error = list()
alpha_origin = list()
phi_origin = list()
phi_set = list()
L2HC = list()
L2MC = list()
alpha_buffer = list()
phi_buffer = list()
log_format = "{} {}"

# [TODO]: Check if arange does not change the output
# for t in range(0.0, tf, h):
v = zeros([5,5]) # array for evolution functions
for t in arange(0.0, tf, h):
  for step in range(4):
    Chi = dot(c0 + v[step,0], psi)
    rChi = dot(c0 + v[step,0], rpsi)
    rrChi = dot(c0 + v[step,0], rrpsi)
    Phi = dot(a0 + v[step,1], psi)
    rPhi = dot(a0 + v[step,1], rpsi)
    rrPhi = dot(a0 + v[step,1], rrpsi)
    Alpha = 1 + dot(al0 + v[step,2], psi)
    rAlpha = dot(al0 + v[step,2], rpsi)
    rrAlpha = dot(al0 + v[step,2], rrpsi)
    Pi = dot(a0 + v[step,3], psi)
    rPi = dot(a0 + v[step,3], rpsi)
    K = dot(fk0 + v[step,4], SB1)
    rK = dot(fk0 + v[step,4], rSB1)

    vx = exp(4*Chi)
    vz = exp(-4*Chi)

    # rhsk * inverted beta matrix
    ck0 = dot((1 + 2*r*rChi)*vx*K/r + vx*rK - Pi*rPhi*vx, inv(2*rChi*SB1 + rSB1 + 3/r*SB1))
    Krr = dot(ck0, SB1)
    rKrr = dot(ck0, rSB1)

    # rhsbe * inverted beta matrix
    be0 = dot(3/2*Alpha*vz*Krr - 1/2*Alpha*K, inv(rSB2 - SB2/r))
    Beta = dot(be0, SB2)
    rBeta = dot(be0, rSB2)
      
    dal = dot(epsilon0*(rrAlpha+rAlpha*(2/r+2*rChi))*vz - epsilon0*Alpha*(1.5*exp(-8*Chi)*Krr**2+0.5*K**2-vz*K*Krr)
          - epsilon0*Alpha*Pi**2 - epsilon0*Beta*rK - epsilon0*eta0*K,inv_psi)
    db = dot(Beta*rPi + vz*rPhi*(2*Alpha/r + 2*Alpha*rChi) + vz*(Alpha*rrPhi + rAlpha*rPhi) + Alpha*K*Pi, inv_psi)
    dc = dot(1/2*rBeta + Beta*rChi - 1/2*Alpha*vz*Krr, inv_psi)
    da = dot(Alpha*Pi + Beta*rPhi, inv_psi)
    dfk = dot(-eta0*K, inv_SB1)


    d = 2 if (step == 1 or step == 2) else 1
    m = lambda x: (h*x)/d 
    v[step+1,:] = m(dal), m(db), m(dc), m(da), m(dfk)

    if step == 0:
      # Hamiltonian constraint L2 error
      qPhi = dot(a0, qpsi)
      rqPhi= dot(a0, rqpsi)
      qPi = dot(b0, qpsi)
      qChi = dot(c0, qpsi)
      rqChi = dot(c0, rqpsi)
      rrqChi = dot(c0, rrqpsi)
      qKrr = dot(ck0, qSB1)
      qK = dot(fk0, qSB1)
  
      vx = exp(4*qChi)
      vz = exp(-4*qChi)
  
      H = 4*rqChi**2 + 4*rrqChi + 8/rq*rqChi + 3/4*vz*qKrr**2 - 1/4*vx*qK**2 + 1/2*vx*(qPi**2 + vz*rqPhi**2) - qK*qKrr/2
      L2 = (1/2*dot(H**2,wq_col))**1/2
      L2HC.append(L2) # L2 error of HC
  
      # Alpha origin
      alpha_0 = 1 + dot(al0, psi_0)
      alpha_origin.append(alpha_0) # = Alphacenter in matlab
      alpha_buffer.append(log_format.format(t,alpha_0))
  
      # Phi origin
      phi_0 = dot(a0, psi_0)
      phi_origin.append(phi_0)
      phi_buffer.append(log_format.format(t,phi_0))
  
      # Error ADM mass
      Madm = 2*dot(arange(1,2*N+2,2),c0)
      Madm_error.append(abs(Madm - M0)/M0 * 100)
  
  print(f'max(al0) = {max(al0)}' )

  # Evolution functions
  a0  = filter * (a0 + 1/6 * (v[1,1] + 2*v[2,1] + 2*v[3,1] + v[4,1])) # L
  b0  = filter * (b0 + 1/6 * (v[1,3] + 2*v[2,3] + 2*v[3,3] + v[4,3])) # N
  c0  = filter * (c0 + 1/6 * (v[1,0] + 2*v[2,0] + 2*v[3,0] + v[4,0])) # K
  al0 = al0          + 1/6 * (v[1,2] + 2*v[2,2] + 2*v[3,2] + v[4,2])  # M
  fk0 = fk0          + 1/6 * (v[1,4] + 2*v[2,4] + 2*v[3,4] + v[4,4])  # P

  phi_set.append(dot(a0, psiplot))

t1 = linspace(0, tf, len(alpha_origin))
tl = f"$A_0$ = {A0}"

'''

with open('alpha_origin.txt','w') as file_alpha:
    file_alpha.truncate(0)
    file_alpha.write("\n".join(alpha_buffer))
with open('phi_origin.txt','w') as file_phi:
    file_phi.truncate(0)
    file_phi.write("\n".join(phi_buffer))

# Searching for critical amplitude
plot_data(t1, alpha_origin, color="g", xlim_a=0, xlim_b=12,
        xlb="t", ylb="$\\alpha(t,0)$", lbl=tl, has_legend=True)

# Phi origin
plot_data(t1, phi_origin, color="b", xlb="t", ylb="$\\phi$(t,0)", lbl=tl,
        has_legend=True, ttl=f"Phi na origem para $L0 = {L0} e N = {N}")

# Hamiltonian constraint L2 error
plot_data(t1, L2HC, xlb="t", ylb="log(L2HC)", lbl=tl,
        ttl=f"log(L2HC) para $N = {N}$, $L_0 = {L0}$ e $A_0 = {A0}$", log_y=True)

# Momentum constraint L2 error
plot_data(t1, L2MC, xlb="t", ylb="log(L2MC)", lbl=tl,
        ttl=f"log(L2MC) para $N = {N}$, $L_0 = {L0}$ e $A_0 = {A0}", log_y=True)

# x = rx
# y = phi_set
def plot_results(x,y,M,figsize,animation_name):
    def generate_animation(interval, xlim=(0,10), ylim=(-1.5,1.5)):
        fig = figure()
        ax = axes(xlim)
        ax.plot([], [], lw=2)

        return FuncAnimation(
                fig,
                partial(_,y=phi_set[i]),
                frames=int(tf/h),
                interval=interval,
                blit=True)

    y = y[0000]
    theta = linspace(0, 2*pi, M) # Revolution of f(phi,r)

    xn = outer(x, cos(theta))
    yn = outer(x, sin(theta))
    zn = zeros_like(xn)
    for i in range(len(x)):
      zn[i,:] = full_like(zn[0,:], y[i])

    fig = figure(figsize)
    ax1 = fig.add_subplot(121, adjustable='box')  # Adjust proportions as needed
    ax1.plot(x, y)
    ax1.grid()
    ax1.set_xlabel('r')
    ax1.set_ylim(-1.5, 1)
    ax1.set_ylabel('$\\phi_j(t,r)$')
    ax1.set_box_aspect(1)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(xn, yn, zn, rstride = 5, cstride = 5, cmap = 'magma', edgecolor = 'none')
    ax2.set_xlabel('$r$')
    ax2.set_ylabel('$r$')
    ax2.set_zlim(-1.5, 1)
    ax2.grid(False)
    subplots_adjust(wspace=0.1)
    show()

    anim = generate_animation(ylim=(-2,1.5), interval=0.2)
    anim.save(animation_name)
'''
