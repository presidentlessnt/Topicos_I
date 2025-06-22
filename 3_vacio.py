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

t  = 2.54		# Espesor cámara de vacío [cm]
dt = .005 		# Incertidumbre espesor cámara de vacío [cm]
dP =   10 		# Incertidumbre presión [mbar]
dN = 1			# Incertidumbre conteo de franjas
E1 = np.array([[100, 200, 300, 400, 500, 600, 700],
	       [  2,   5,   7,   9,  11,  13,  15]]).T 	# ΔP [mbar], N


odr_lamb  = 631.2143672631	# [nm]
odr_dlamb =   2.7876510585
mpe_lamb  = 633.9085889736
mpe_dlamb =  18.3933750772

lamb  =  odr_lamb*1e-3 		# [μm]
dlamb = odr_dlamb*1e-3 		# [μm]



# =============================================
# PREPARAR DATOS PARA TRABAJAR CON ODR
# =============================================

D1=np.zeros((len(E1),4))

D1[:,0] = E1[:,0]				# Valores x-axis ΔP [mbar]
D1[:,1] = E1[:,1] * lamb/(2 * t)*1e-4		# Valores y-axis N*lambda/(2*t) -> [μm]/[cm] <> 1e-4
D1[:,2] = E1[:,0] * 0 + dP 			# Error x-axis  ΔP [mbar]
D1[:,3] = E1[:,1] * (lamb/(2*t)*1e-4) * np.sqrt( (dN/E1[:,1])**2 + (dlamb/lamb)**2 + (dt/t)**2 ) 	# Error y-axis N*lambda/(2*t)



# Lineal function
def func(p,x):
    b,c = p
    return b*x+c


# Model object
quad_model = odr.Model(func)


# Create a RealData object
data = odr.RealData(D1[:,0], D1[:,1], sx=D1[:,2], sy=D1[:,3])


# Set up ODR with the model and data.
odr = odr.ODR(data, quad_model, beta0=[2., .1])

# Run the regression.
out = odr.run()

#print fit parameters and 1-sigma estimates
popt = out.beta
perr = out.sd_beta


slope   = popt[0]*1e7			# N*lambda/(2*t*ΔP) [mbar^-1]
pmslope = perr[0]*1e7			# N*lambda/(2*t*ΔP) [mbar^-1]

slope_Hg   = popt[0]*1e7 * 13.332	# [mbar]/[cmHg]
pmslope_Hg = perr[0]*1e7 * 13.332	# [mbar]/[cmHg]

# prepare confidence level curves

nstd = 5. # to draw 5-sigma intervals
popt_up = popt + nstd * perr
popt_dw = popt - nstd * perr

x_fit = D1[:,0] 
fit = func(popt, x_fit)
fit_up = func(popt_up, x_fit)
fit_dw = func(popt_dw, x_fit)



# =============================================
# PREPARAR DATOS PARA TRABAJAR CON MPE
# =============================================

D2 = np.zeros((len(E1),2))

D2[:,0]  = E1[:,1] * (lamb/(2*t*E1[:,0])*1e3)
D2[:,1]  = D2[:,0] * np.sqrt( (dN/E1[:,1])**2 + (dlamb/lamb)**2 + (dt/t)**2 + (dP/E1[:,0])**2 )
cslope   = np.mean(D2[:,0])
cpmslope = np.sqrt(np.sum(D2[:,1]**2))/len(D2)
cslope_Hg   = cslope * 13.332
cpmslope_Hg = cpmslope * 13.332

# PARA CALCULO DE n_aire A 1ATM

P_1atm  = 1013.25	# [mbar]
n_vacio = 1
n_1atm  = slope*1e-7 * P_1atm + n_vacio		# ODR
cn_1atm  = cslope*1e-7 * P_1atm + n_vacio	# MPE




# =============================================
# RESULTADOS EN CONSOLA
# =============================================

print("\n" + "="*60)
print("ÍNDICE DE REFRACCIÓN DEL AIRE EN CÁMARA DE VACÍO".center(60))
print("="*60)
print("MÉTODO ODR:")
print(f"{'Δn/ΔP =':<2}{'':<2}{slope:.4f}{'':<2}± {pmslope:.4f} [·10^-7  1/mbar]")
print(f"{'Δn/ΔP =':<2}{'':<2}{slope_Hg/10:.4f}{'':<2}± {pmslope_Hg/10:.4f} [·10^-6  1/cmHg]")
print("\n" + "-"*60)
print("MÉTODO MPE:")
print(f"{'Δn/ΔP =':<2}{'':<2}{cslope:.4f}{'':<2}± {cpmslope:.4f} [·10^-7  1/mbar]")
print(f"{'Δn/ΔP =':<2}{'':<2}{cslope_Hg/10:.4f}{'':<2}± {cpmslope_Hg/10:.4f} [·10^-6  1/cmHg]")
print("\n" + "="*60)
print("ÍNDICE DE REFRACCIÓN DEL AIRE [ODR] A: 1 atm <> 1013.25 mbar")
print(f"n_1atm ={'':<10} Δn/ΔP {'':<9}* {'':<3} P_1atm {'':<4}+ n_vacío")
print(f"{'n_1atm =':<2}{'':<2}{slope:.4f} [·10^-7  1/mbar] * {P_1atm:.2f} [mbar] +{'':<2}{n_vacio:.4f}")
print(f"{'n_1atm =':<2}{'':<2}{n_1atm:.6f}")
print("\n" + "-"*60)
print("ÍNDICE DE REFRACCIÓN DEL AIRE [MPE] A: 1 atm <> 1013.25 mbar")
print(f"n_1atm ={'':<10} Δn/ΔP {'':<9}* {'':<3} P_1atm {'':<4}+ n_vacío")
print(f"{'n_1atm =':<2}{'':<2}{cslope:.4f} [·10^-7  1/mbar] * {P_1atm:.2f} [mbar] +{'':<2}{n_vacio:.4f}")
print(f"{'n_1atm =':<2}{'':<2}{cn_1atm:.6f}")
print("\n" + "="*60)
#print(f"Tabla 3")
#print(np.array2string(D2, formatter={'float_kind': lambda x: "%.4f" % x}))


# =============================================
# RESULTADOS GRÁFICOS - PLOTEO
# =============================================
plt.style.use('seaborn-v0_8-notebook')
plt.rcParams.update({'font.size': 12})

plt.figure(1)
plt.errorbar(D1[:,0], D1[:,1], xerr=D1[:,2], yerr=D1[:,3], ecolor='k', fmt='none', label='Data')
plt.xlabel('$P\;[mbar]$')
plt.ylabel('$\dfrac{N*\lambda}{2*d}$')

plt.plot(x_fit, fit, 'r', lw=2, label='Ajuste lineal \n $\dfrac{\Delta n}{\Delta P}:%1.2f \pm %1.2f\;[\cdot 10^{-7}\,mbar^{-1}]$'%(slope,pmslope))
plt.fill_between(x_fit, fit_up, fit_dw, alpha=.25, label='Intervalo 5-sigma')
plt.legend(loc=4)
plt.grid(ls='--',color='grey',lw=.5)
plt.ylim(0,250e-6)
plt.xlim(50,750)
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0), useMathText=True)
plt.title(f'Indice refracción del aire: $n$')
plt.tight_layout()  # Mejor ajuste de los elementos

plt.show()
