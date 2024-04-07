# IMPORTS

#from mpl_toolkits.mplot3d import Axes3D
#from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from functools import partial
from copy import copy
from numpy import pi, arange, arctan, dot, exp, flip, sin, cos, sqrt, linspace, outer, ones, zeros, zeros_like, full, full_like
from numpy.polynomial.legendre import leggauss
from numpy.linalg import inv

import matplotlib.pyplot as plt

# RECURRING FUNCTIONS
# [TODO]: use adequate func names
def chebyshev(N,M,r,method="alpha"):
    y = zeros([N,M])
    dy = copy(y)
    ddy = copy(y)

    for i in range(N):
      (a,b) = (2*i+1,2*i+1) if method == "alpha" else (2*(i+0.5)+1,2*i+2)
      f = sin(a*rarct)
      d = cos(b*rarct)*L0
      c = -sin(b*rarct)*b**2
      e = (1+L0**2/r**2)
      y[i,] = f
      dy[i,] = -d/(r**2*e)
      ddy[i,] = c*L0**2/(r**4*e**2)+2*d/(r**3*e)-2*d**3/(r**5*e**2)
    return (y, dy, ddy)

def alphaphi(ax, sigma):
  sigma2 = sigma**2
  return ax * (exp(-(r-r0)**2/sigma2)+exp(-(r+r0)**2/sigma2))

# BLOCK 1

N = 200 # Truncation order
L0 = 2 # Map parameter

colr = cos(arange(2*N+4)*pi/(2*N+3))[1:N+2] # collocation points (Verificado)
r1 = L0 * colr/(sqrt(1-colr**2)) # physical domain (Verificado)
r = flip(r1)
rarct = arctan(L0/r)

# Base Matrix (Chebyshev Polinomials):
SB, rSB, rrSB = chebyshev(N+3,N+1,r)
psi, rpsi, rrpsi = SB[0:N+1,:], rSB[0:N+1,:], rrSB[0:N+1,:]

# Initial conditions of Phi (Scalar field)

r0 = 2
sigma = 1
A0 = 0.1 # 0.09 -> disperses ; 0.09 -> collapses

Phi_0 = A0*r**2*(exp(-(r-r0)**2/sigma**2)+exp(-(r+r0)**2/sigma**2)) # Phi initial data (Verificado)
inv_psi = inv(psi)
a0 = dot(Phi_0, inv_psi) # coeficients a(0)  (Verificado)
Phi = dot(a0, psi) # approximate solution in t = 0
rPhi= dot(a0, rpsi)

########################### Plot: Initial conditions for Phi

M = 3000 # plot truncation

rplot = linspace(0.000001,10,M)
colplot = rplot/sqrt(L0**2 + rplot**2)
SBplot, rSBplot, rrSBplot = chebyshev(N+1,M,rplot)
psiplot, rpsiplot, rrpsiplot = SBplot[0:N+1,:], rSBplot[0:N+1,:], rrSBplot[0:N+1,:]
Phi_plot0 = alphaphi(A0*rplot**2, sigma) 
Phiplot = dot(a0, psiplot)

plt.plot(rplot, Phiplot, rplot, Phi_plot0, "--r") #(Verificado)
plt.xlabel('r')
plt.xlim(0,8)
plt.grid()
plt.show()

# BLOCK 2

erro = Phi_plot0 - Phiplot
plt.plot(rplot, erro)
plt.xlabel('r')
plt.xlim(0,8)
plt.ylabel("$|\\phi_N - \\phi_0|$")
plt.grid()
plt.show()

# BLOCK 3

# Initial conditions for Alpha:

al0 = dot(alphaphi(1-A0, sigma) - 1, inv_psi)

# Initial values of Krr and K:

# Base functions:

SB1 = 1/2*(SB[1:(N+2),:] + SB[0:(N+1),:]) # Verificado
rSB1 = 1/2*(rSB[1:(N+2),:] + rSB[0:(N+1),:])
rrSB1 = 1/2*(rrSB[1:(N+2),:] + rrSB[0:(N+1),:])

inv_SB1 = inv(SB1)
K0 = A0/20*r**2*(exp(-(r)**2/sigma**2) + exp(-(r)**2/sigma**2))
fk0 = dot(K0, inv_SB1)
K = dot(fk0, SB1)

b0 = dot(zeros(N+1), psi)
Pi = dot(b0, psi)
c0 = full(N+1, 0.001)

# Newton-Raphson loop:

tol = 1e-18 # tolerance
iterations = 50
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
  JH = 8*J1 + 4*rrpsi + 8/r*rpsi + 4*psi*vx*K/4 - 2*psi*vw + 1/2*vx*(4*psi*vz*rPhi2)
  inv_JH = inv(JH)
  c = c0 - dot(H0, inv_JH)
  if max(abs(c0 - c)) <= tol:
    break
  c0 = c

# BLOCK 4

Chiplot = dot(c0, psiplot)
rrChiplot = dot(c0, rrpsiplot)

plt.plot(rplot,Chiplot)
plt.xlim(0,8)
plt.grid()
plt.show()

# BLOCK 5

# Field Equations

# Base functions for Beta

SB2, rSB2, rrSB2 = chebyshev(N+1,N+1,r,method="beta")

# BLOCK 6

# Quadrature Integration

# Legendre quadrature points
Nq = int(3/2*N) # Quadrature truncation
new_col = leggauss(Nq+1)[0]

# Legendre Polinomials
P = zeros([Nq+3,Nq+1])
P[0,] = 1
P[1,] = new_col
colP = copy(P)
colP[1,] = 1

for i in range(2,Nq+3):
  P[i,] = ((2*i-1)*new_col*P[i-1,] - (i-1)*P[i-2,])/(i)
  colP[i,] = i*P[i-1] + new_col*colP[i-1]

P_max = P[Nq+1]
colP_max = colP[Nq+1]
wq_col = 2/((1-new_col**2)*colP_max**2) # Legendre weight (Verificado)
rq = L0*(1+new_col)/(1-new_col) # Physical quadrature domain

qSB, qrSB, qrrSB = chebyshev(Nq+3,Nq+1,L0,rq)

qpsi = qSB[0:N+1,:]
rqpsi = qrSB[0:N+1,:]
rrqpsi = qrrSB[0:N+1,:]

# Initial Phi in quadrature points
#qPhi = dot(a0, qpsi)
#rqPhi= dot(a0, rqpsi)

# Initial Pi for quadrature points
#qPi = dot(b0, qpsi)

# Initial Chi for quadrature points:
#qChi = dot(c0, qpsi)   # Verificado todos
#rqChi = dot(c0, rqpsi)
#rrqChi = dot(c0, rrqpsi)

qSB1 = 1/2*(qSB[1:(N+2),:] + qSB[0:(N+1),:]) # Verificado
rqSB1 = 1/2*(qrSB[1:(N+2),:] + qrSB[0:(N+1),:])
rrqSB1 = 1/2*(qrrSB[1:(N+2),:] + qrrSB[0:(N+1),:])
#qKrr = dot(ck0, qSB1)

# Alpha na origem

psi_0 = zeros(N+1) # psi(t,0)
for i in range(N+1):
  psi_0[i,] = sin((2*i+1)*pi/2) # arccot(0) = Pi/2

# BLOCK 7

# Filtering

#Nc = 0
#Nf = N - Nc
#coef_f = 36
#s = 10
#filter1 = exp(- coef_f*((arange(N - Nc + 1))/(N-Nc))**s)

filter1 = ones(N+1)

# BLOCK 8

# Runge-Kutta 4th order

h = 0.0002 # step size
t = 0
tf = 7.0
epsilon0 = 1
eta0 = 1
V = 0

M0 = 2*dot(arange(1,2*N+2,2),c0) # Madm(t = 0)

Madm_error = list()
Alpha_origin = list()
phi_origin = list()
L2HC = list()
L2MC = list()
phi_set = list()
alpha_buffer = list()
phi_buffer = list()
log_format = "{} {}"

# [TODO]: Check if arange does not change the output
# for t in range(0.0, tf, h):
v = zeros([5,5])
for t in arange(0.0, tf, h):
  for step in range(4):
    Phi = dot(a0 + v[step,1], psi)
    rPhi = dot(a0 + v[step,1], rpsi)
    rrPhi = dot(a0 + v[step,1], rrpsi)
    Pi = dot(a0 + v[step,3], psi)
    rPi = dot(a0 + v[step,3], rpsi)
    Chi = dot(c0 + v[step,0], psi)
    rChi = dot(c0 + v[step,0], rpsi)
    rrChi = dot(c0 + v[step,0], rrpsi)
    Alpha = 1 + dot(al0 + v[step,2], psi)
    rAlpha = dot(al0 + v[step,2], rpsi)
    rrAlpha = dot(al0 + v[step,2], rrpsi)
    K = dot(fk0 + v[step,4], SB1)
    rK = dot(fk0 + v[step,4], rSB1)

    vx = exp(4*Chi)
    vz = exp(-4*Chi)

    Matrix_Krr = 2*rChi*SB1 + rSB1 + 3/r*SB1   
    inv_matrix_krr = inv(Matrix_Krr)
    rhsk = (1 + 2*r*rChi)*vx*K/r + vx*rK - Pi*rPhi*vx
    ck0 = dot(rhsk, inv_matrix_krr)
    Krr = dot(ck0, SB1)
    rKrr = dot(ck0, rSB1)

    Matrix_Beta = rSB2 - SB2/r
    inv_matrix_beta = inv(Matrix_Beta)
    rhsbe = 3/2*Alpha*vz*Krr - 1/2*Alpha*K
    be0 = dot(rhsbe, inv_matrix_beta)
    Beta = dot(be0, SB2)
    rBeta = dot(be0, rSB2)
      
    dal = dot(epsilon0*(rrAlpha + rAlpha*(2/r + 2*rChi))*vz - epsilon0*Alpha*(1.5*exp(-8*Chi)*Krr**2 + 0.5*K**2 - vz*K*Krr) - epsilon0*Alpha*Pi**2 - epsilon0*Beta*rK - epsilon0*eta0*K, inv_psi)
    db = dot(Beta*rPi + vz*rPhi*(2*Alpha/r + 2*Alpha*rChi) + vz*(Alpha*rrPhi + rAlpha*rPhi) + Alpha*K*Pi, inv_psi)
    dc = dot(1/2*rBeta + Beta*rChi - 1/2*Alpha*vz*Krr, inv_psi)
    da = dot(Alpha*Pi + Beta*rPhi, inv_psi)
    dfk = dot(-eta0*K, inv_SB1)

    # [TODO] is there a reason for these parentheses?
    match step:
      case 0:
        v[1:,0] = [h*(dc), h*(da), h*(dal), h*(db), h*(dfk)]
      case 1:
        v[2:,0] = [(h*(dc))/2, (h*(da))/2, (h*(dal))/2, (h*(db))/2, (h*(dfk))/2]
      case 2:
        v[3:,0] = [(h*(dc))/2, (h*(da))/2, (h*(dal))/2, (h*(db))/2, (h*(dfk))/2]
      case 3:
        v[4:,0] = [h*(dc), h*(da), h*(dal), h*(db), h*(dfk)]

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
      Alpha_0 = 1 + dot(al0, psi_0)
      Alpha_origin.append(Alpha_0) # = Alphacenter in matlab
      alpha_buffer.append(log_format.format(t,Alpha_0))
  
      # Phi origin
      phi_0 = dot(a0, psi_0)
      phi_origin.append(phi_0)
      phi_buffer.append(log_format.format(t,phi_0))
  
      # Error ADM mass
      Madm = 2*dot(arange(1, 2*N + 2, 2), c0)
      Madm_error.append(abs(Madm - M0)/M0 * 100)
  
  print(f'max(al0) = {max(al0)}' )

  # Evolution functions
  a0 = filter1 * (a0 + 1/6 * (v[1,1] + 2*v[2,1] + 2*v[3,1] + v[4,1]))
  b0 = filter1 * (b0 + 1/6 * (v[1,3] + 2*v[2,3] + 2*v[3,3] + v[4,3]))
  c0 = filter1 * (c0 + 1/6 * (v[1,0] + 2*v[2,0] + 2*v[3,0] + v[4,0]))
  al0 = al0          + 1/6 * (v[1,2] + 2*v[2,2] + 2*v[3,2] + v[4,2])
  fk0 = fk0          + 1/6 * (v[1,4] + 2*v[2,4] + 2*v[3,4] + v[4,4])

  phi_set.append(dot(a0, psiplot))

t1 = linspace(0, tf, len(Alpha_origin))

out_a = open('alpha_origin.txt', 'w')
out_p = open('phi_origin.txt', 'w')
out_a.truncate(0) # erase old data
out_p.truncate(0)
out_a.write("\n".join(alpha_buffer))
out_p.write("\n".join(phi_buffer))

# BLOCK 9

# Searching for critical amplitude

Alpha_origin_disp = Alpha_origin
#Alpha_origin_collapse = Alpha_origin

plt.plot(t1, Alpha_origin_disp, color = "g", label = f'$A_0$ = {A0}')
#plt.plot(t1, Alpha_origin_collapse, color = "y", label = "$A_0$ = {:}".format(A0))
plt.ylabel(r"$\alpha(t,0)$")
plt.xlabel("t")
plt.xlim(0,12)
plt.grid()
plt.legend()
plt.show()

# BLOCK 10

# Phi origin

plt.plot(t1, phi_origin, color = "b", label = " = {:}".format(A0))
plt.title("Phi na origem para L0 = 2 e N = {:}".format(N) )
plt.ylabel("$\\phi$(t,-1)")
plt.xlabel("t")
plt.grid()
plt.legend()

# BLOCK 11

# Hamiltonian constraint L2 error

plt.plot(t1,L2HC)
plt.yscale("log")
plt.ylabel("log(L2HC)")
plt.xlabel("t")
plt.grid()
plt.title("log(L2Hc) para $N = 200$, $L_0 = 20$ e $A_0 = $")

# BLOCK 11

# Momentum constraint L2 error

plt.plot(t1,L2MC,label = "$A_0$ = {:}".format(A0))
plt.yscale("log")
plt.ylabel("log(L2HC)")
plt.xlabel("t")
plt.grid()
plt.title("log(L2MC) para $N = 50$, $L_0 = 2$")

# K-Driver slicing results: dispersion

# BLOCK 12

# Scalar field 3D plot at t = constant: Dispersion case

fig = plt.figure(figsize=(12,6))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122,projection='3d')

y = phi_set[0000] # 5 plots from t = 0, .., 10
x = rplot
theta = linspace(0, 2*pi, M) # Revolution of f(phi,r)

xn = outer(x, cos(theta))
yn = outer(x, sin(theta))
zn = zeros_like(xn)

for i in range(len(x)):
  zn[i,:] = full_like(zn[0,:], y[i])

#ax2.plot_surface(xn, yn, zn)
ax2.plot_surface(xn, yn, zn, rstride = 5, cstride = 5, cmap = 'magma', edgecolor = 'none')
ax1.plot(x, y)
ax1.grid()
ax1.set_xlabel('r')
ax1.set_ylim(-1.5, 1)
ax1.set_ylabel('$\\phi_j(t,r)$')
#ax1.set_aspect('equal')
ax1.set_box_aspect(1)
ax2.set_xlabel('$r$')
ax2.set_ylabel('$r$')
ax2.set_zlim(-1.5, 1)
#ax2.set_zlabel('$\phi$')

plt.subplots_adjust(wspace=0.1)

ax2.grid(False)
#plt.axis('off')
plt.show()

# BLOCK 13

def init():
  line.set_data([], [])
  initA0_text.set_text('')
  time_text.set_text('')
  return line,

def animate(i,y):
  line.set_data(x, y)
  initA0_text.set_text("$A_0 = {:}$".format(A0))
  time_text.set_text("Time ="+str(round(h+h*i,2)))
  return line,

def generate_animation(xlim=(0,10), ylim=(-1.5,1.5), interval=0.05):
    fig = plt.figure()
    ax = plt.axes(xlim, ylim)
    line, = ax.plot([], [], lw=2)
    initA0_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
    time_text = ax.text(0.02, 0.9, '', transform=ax.transAxes)
    x = rplot

    return FuncAnimation(fig, partial(animate,y=phi_set[i]), init_func=init, frames=int(tf/h), interval, blit=True)

# Animation plot for Phi: Scalar Field dispersion

anim = generate_animation()
anim.save("animation_KD_dispersion.mp4")

# K-Driver slicing results: collapse

# BLOCK 14

# Scalar field 3D plot at t = constant: Collapse case

fig = plt.figure(figsize=(8,4))
ax1 = fig.add_subplot(121, adjustable='box')  # Adjust proportions as needed
ax2 = fig.add_subplot(122, projection='3d')

M = 3000

y = phi_set[0000] # 7 plots from t = 0, 2, 4, ..., 12
x = rplot
theta = linspace(0, 2*pi, M) # Revolution of f(phi,r)

xn = outer(x, cos(theta))
yn = outer(x, sin(theta))
zn = zeros_like(xn) # [TODO] check if this can be simplified

for i in range(len(x)):
  zn[i,:] = full_like(zn[0,:], y[i])

#ax2.plot_surface(xn, yn, zn)
ax2.plot_surface(xn, yn, zn, rstride = 5, cstride = 5, cmap = 'magma', edgecolor = 'none')
ax1.plot(x, y)
ax1.grid()
ax1.set_xlabel('r')
ax1.set_ylim(-1.5, 1)
ax1.set_ylabel('$\\phi_j(t,r)$')
#ax1.set_aspect('equal')
ax1.set_box_aspect(1)
ax2.set_xlabel('$r$')
ax2.set_ylabel('$r$')
ax2.set_zlim(-1.5, 1)
#ax2.set_zlabel('$\phi$')

#bbox = ax2.get_position()
#ax1.set_position([bbox.x0, bbox.y0, bbox.width, bbox.height])

plt.subplots_adjust(wspace=0.1)

ax2.grid(False)
#plt.axis('off')
plt.show()

# BLOCK 15

# Animation plot for Phi: Scalar Field collapse

#HTML(anim.to_html5_video())
anim = generate_animation(ylim=(-2,1.5), interval=0.2)
anim.save("animation_MS_collapse.mp4")

