#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy import odr
import warnings
warnings.filterwarnings("ignore")


# =============================================
# DATOS DE ENTRADA
# =============================================

t  = 5.0	 		# [mm] Espesor de placa vidrio
dt = .05/2			# [mm] Error medicion espesor (con vernier)
dO = 0.1/2 * np.pi/180	 	# [rad] Error medicion angular
dN = 1				# Error conteo franjas

E1 = np.array([[3.8, 5.0, 6.2, 7.0, 7.8, 8.5, 9.2, 9.6, 10.2, 10.6, 11.2, 11.8, 12.3, 12.8, 13.7, 14.6, 15.4],
	       [ 10,  18,  30,  40,  50,  60,  70,  80,   90,  100,  110,  120,  130,  140,  160,  180,  200]]).T 	# O[deg], N


odr_lamb  = 631.2143672631	# [nm]
odr_dlamb =   2.7876510585
mpe_lamb  = 633.9085889736
mpe_dlamb =  18.3933750772

lamb  =  odr_lamb*1e-3 		# [μm]
dlamb = odr_dlamb*1e-3 		# [μm]



# =============================================
# PREPARAR DATOS PARA TRABAJAR CON ODR
# =============================================

D2 = np.zeros((len(E1),4))

E1[:,0]*=np.pi/180		# Pasar angulos [deg] -> [rad]

## n_g(t, N, theta, lambda) = Num / Den
## Num = (2*t-N*lambda)*(1-cos(theta))
## Den = 2*t*(1-cos(theta)) - N*lambda
## dNum = sqrt{ (1-cos(theta))**2 * [(2*dt)**2 + (lambda*dN)**2 + (N*dlambda)**2] + [(2*t-N*lambda)*sin(theta)*dtheta]**2 }
## dDen = sqrt{ [2*(1-cos(theta))*dt]**2   +   (lambda*dN)**2   +   (2*t*sin(theta)*dtheta)**2   +   (N*dlambda)**2 }

## dn_g = sqrt{ [2*(1-cos(theta))*(Den - Num)*dt]**2   +  [lambda*(1-cos(theta))*Den - lambda*Num]**2 * dN**2  +  ...
##        ...[(2*t-N*lambda)*Den - 2*t*sin(theta)*Num]**2 * dtheta**2  +  [(1-cos(theta))*Den - Num]**2 * (N*dlambda)**2 } / Den**2

t*=1e3		# [cm] -> [μm]
dt*=1e3		# [cm] -> [μm]

for i in range(len(E1)):
    D2[i,0] = 2*t * (1 - np.cos(E1[i,0])) - E1[i,1]*lamb             	# Den [μm]
    D2[i,1] = (2*t - E1[i,1]*lamb) * (1-np.cos(E1[i,0]))             	# Num [μm]
    D2[i,2] = np.sqrt((2*(1-np.cos(E1[i,0]))*dt)**2 + (lamb*dN)**2 + (2*t*np.sin(E1[i,0])*dO)**2 + (E1[i,1]*dlamb)**2) # dDen [μm]
    D2[i,3] = np.sqrt((1 - np.cos(E1[i,0]))**2 * ((2*dt)**2 + (lamb*dN)**2 + (E1[i,1]*dlamb)**2) + ((2*t-E1[i,1]*lamb)*np.sin(E1[i,0])*dO)**2)                        		# dNum [μm]
    


# Lineal function
def func(p, x):
    b, c = p
    return b*x + c

# Model object
quad_model = odr.Model(func)


# Create a RealData object
data = odr.RealData(D2[:,0], D2[:,1], sx=D2[:,2], sy=D2[:,3])

# Set up ODR with the model and data.
odr = odr.ODR(data, quad_model, beta0=[1.5, .1])

# Run the regression.
out = odr.run()

#print fit parameters and 1-sigma estimates
popt = out.beta
perr = out.sd_beta


slope   = popt[0]
pmslope = perr[0]
vodr    = popt[1]
pmvodr  = perr[1]

# prepare confidence level curves
nstd = 5. # to draw 5-sigma intervals
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

x_fit = D2[:,0]
fit = func(popt, x_fit)
fit_up = func(popt_up, x_fit)
fit_dw= func(popt_dw, x_fit)


# =============================================
# PREPARAR DATOS PARA TRABAJAR CON MPE
# =============================================

## ECUACIONES A USAR - NOTACIÓN SIMBÓLICA
## n_g(t, N, theta, lambda) = Num / Den
## Num = (2*t-N*lambda)*(1-cos(theta))
## Den = 2*t*(1-cos(theta)) - N*lambda
## dn_g = sqrt{ [2*(1-cos(theta))*(Den - Num)*dt]**2    +   [lambda*(1-cos(theta))*Den - lambda*Num]**2 * dN**2   +  ...
##        ...[(2*t-N*lambda)*Den - 2*t*sin(theta)*Num]**2 * dtheta**2   +   [(1-cos(theta))*Den - Num]**2 * (N*dlambda)**2 } / Den**2


NG = np.zeros((len(E1),2))

for i in range(len(E1)):
    NG[i,0] = D2[i,1] / D2[i,0]
    da = 2*(1-np.cos(E1[i,0]))*(D2[i,0] - D2[i,1])*dt
    db = (lamb*(1-np.cos(E1[i,0]))*D2[i,0] - lamb*D2[i,1]) * dN
    dc = ((2*t-E1[i,1]*lamb)*D2[i,0] - 2*t*np.sin(E1[i,0])*D2[i,1]) * dO
    dd = ((1-np.cos(E1[i,0]))*D2[i,0] - D2[i,1]) * (E1[i,1]*dlamb)
    NG[i,1] = np.sqrt(da**2 + db**2 + dc**2 + dd**2)/D2[i,0]**2

cslope   = np.mean(NG[:,0])
cpmslope = np.sqrt(np.sum(NG[:,1]**2))/len(NG)



# =============================================
# RESULTADOS EN CONSOLA
# =============================================

print("\n" + "="*40)
print("ÍNDICE DE REFRACCIÓN DEL VIDRIO".center(40))
print("="*40)
print("MÉTODO ODR:")
print(f"{'n_g =':<2}{'':<2}{slope:.4f}{'':<2}± {pmslope:.4f}")
print("\n" + "-"*40)
print("MÉTODO MPE:")
print(f"{'n_g =':<2}{'':<2}{cslope:.4f}{'':<2}± {cpmslope:.4f}")
print("\n" + "="*40)
#print(f"Tabla 4")
#print(np.array2string(NG, formatter={'float_kind': lambda x: "%.4f" % x}))


# =============================================
# RESULTADOS GRÁFICOS - PLOTEO
# =============================================
plt.style.use('seaborn-v0_8-notebook')
plt.rcParams.update({'font.size': 12})

plt.figure(1)
plt.errorbar(D2[:,0], D2[:,1], yerr=D2[:,3], xerr=D2[:,2], ecolor='k', fmt='none', label='Datos')
plt.ylabel(r'$(2t-N\lambda_0)(1-\cos\theta)\;[\mu m]$')
plt.xlabel(r'$2t(1-\cos\theta)-N\lambda_0\;[\mu m]$')

plt.plot(x_fit, fit, 'r', lw=2, label=f'Ajuste lineal \n $n_g$: {slope:1.4f} $\pm$ {pmslope:1.4f}')
#plt.plot(x_fit, x_fit*cslope+vodr, 'g', lw=2, label=f'Ajuste lineal \n $n_g$: {cslope:1.4f} $\pm$ {cpmslope:1.4f}')
plt.fill_between(x_fit, fit_up, fit_dw, alpha=.25, label='Intervalo 5-sigma')
plt.legend(loc=4)
plt.grid(ls='--', color='grey', lw=.5)
plt.ylim(0, 4.0e2)
plt.xlim(0, 2.5e2)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0), useMathText=True)
plt.title(f'Indice refracción del vidrio: $n_g$')
plt.tight_layout()  # Mejor ajuste de los elementos


plt.figure(2)
plt.plot(E1[:,0]*180/np.pi, NG[:,0], 'o-', label='Datos')
plt.plot(E1[:,0]*180/np.pi, E1[:,0]*0+slope, 'r--', lw=2, label=f'Ajuste lineal ODR: {slope:1.4f} $\pm$ {pmslope:1.4f}')
plt.plot(E1[:,0]*180/np.pi, E1[:,0]*0+cslope, 'g--', lw=2, label=f'Ajuste lineal MPE: {cslope:1.4f} $\pm$ {cpmslope:1.4f}')
plt.ylabel('$n_g$')
plt.xlabel(r'$\theta\;[deg]$')
plt.grid(ls='--', color='grey', lw=.5)
plt.ylim(1.4, 1.6)
plt.xlim(2.0, 18.)
plt.legend(loc=4)
plt.title(f'Indice refracción del vidrio: $n_g$')
plt.tight_layout()  # Mejor ajuste de los elementos

plt.show()

