#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import odr
import warnings
warnings.filterwarnings("ignore")

# =============================================
# DATOS EXPERIMENTALES - FABRY-PEROT Y MICHELSON
# =============================================

# Datos Fabry-Perot
dD_fp = 0.5  # [μm]
dN_fp = 1
E1_fp = np.array([[11, 21, 31, 41, 51, 61, 71, 81, 91, 101, 111, 121, 131, 141, 151, 161, 171, 181, 191, 201],
                   [4,  7, 10, 13, 16, 20, 23, 26, 30,  33,  36,  39,  42,  45,  49,  51,  55,  58,  61,  64]]).T

# Datos Michelson
dD_mi = 0.5  # [μm]
dN_mi = 1
E1_mi = np.array([[20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
                   [6,  9, 12, 15, 19, 22, 25, 28,  31,  34,  38,  40,  44,  47,  50,  53,  56,  59,  62]]).T

# =============================================
# FUNCIÓN PARA PROCESAR DATOS
# =============================================

def procesar_datos(E1, dD, dN, nombre):
    # Preparar matriz ODR
    D1 = np.zeros((len(E1),4))
    D1[:,0] = E1[:,0]      	 # N
    D1[:,1] = E1[:,1]*2    	 # 2D
    D1[:,2] = E1[:,0]*0 + dN 	 # Error N
    D1[:,3] = E1[:,0]*0 + dD*2	 # Error 2D
    
    # Ajuste ODR
    def func(p, x):
        return p[0]*x + p[1]
    
    quad_model = odr.Model(func)
    data = odr.RealData(D1[:,0], D1[:,1], sx=D1[:,2], sy=D1[:,3])
    odr_fit = odr.ODR(data, quad_model, beta0=[0.6, 0.0])
    out = odr_fit.run()
    
    # Resultados ODR
    lambda_odr = out.beta[0] * 1e3  # en nm
    error_odr = out.sd_beta[0] * 1e3
    c_odr = out.beta[1]
    error_c_odr = out.sd_beta[1]
    
    # Cálculo MPE
    D2 = np.zeros((len(E1),2))
    for i in range(len(E1)):
        D2[i,0] = E1[i,1]*2/E1[i,0]
        D2[i,1] = np.sqrt((2/E1[i,0]*dD)**2+(E1[i,1]*2/E1[i,0]**2*dN)**2)
    lambda_classic = np.mean(D2[:,0]) * 1e3
    delta_lambda   = np.sqrt(np.sum(D2[:,1]**2))/len(D2) * 1e3
    
    return {
        'nombre': nombre,
        'D1': D1,
        'D2': D2*1e3,
        'odr_params': (lambda_odr, error_odr, c_odr, error_c_odr),
        'classic_params': (np.mean(lambda_classic), np.mean(delta_lambda)),
        'raw_data': (lambda_classic, delta_lambda)
    }

# =============================================
# PROCESAMIENTO DE DATOS
# =============================================

fp_data = procesar_datos(E1_fp, dD_fp, dN_fp, "Fabry-Perot")
mi_data = procesar_datos(E1_mi, dD_mi, dN_mi, "Michelson")

# ODR
odr_lamb_mi, odr_dlamb_mi = mi_data['odr_params'][:2]
odr_lamb_fp, odr_dlamb_fp = fp_data['odr_params'][:2]
ol  = (odr_lamb_mi + odr_lamb_fp)/2
odl = (odr_dlamb_mi + odr_dlamb_fp)/2

# MPE
cls_lamb_mi, cls_dlamb_mi = mi_data['classic_params']
cls_lamb_fp, cls_dlamb_fp = fp_data['classic_params']
cl  = (cls_lamb_mi + cls_lamb_fp)/2
cdl = (cls_dlamb_mi + cls_dlamb_fp)/2


# =============================================
# RESULTADOS COMPARATIVOS
# =============================================

print("\n" + "="*60)
print("INTERFEROMETRÍA - LONGITUD DE ONDA".center(60))
print("="*60)

for data in [mi_data, fp_data]:
    print(f"\n{data['nombre']}")
    print("-"*60)
    print("MÉTODO ODR: Orthogonal Distance Regression")
    print(f"• λ: {data['odr_params'][0]:.2f} ± {data['odr_params'][1]:.2f} nm")
    #print(f"• c: {data['odr_params'][2]:.2f} ± {data['odr_params'][3]:.2f} μm")
    print("\nMÉTODO MPE: Media con Propagación de Errores")
    #print(np.array2string(data['D2'], formatter={'float_kind': lambda x: "%.2f" % x}))
    print(f"• λ: {data['classic_params'][0]:.2f} ± {data['classic_params'][1]:.2f} nm")

print("\n" + "-"*60)
print("Promediando resultados de los dos Interferómetros por método")
print(f"• λ_ODR: {ol:.2f} ± {odl:.2f} nm")
print(f"• λ_MPE: {cl:.2f} ± {cdl:.2f} nm")
#print(f"{ol:.10f}")
#print(f"{odl:.10f}")
#print(f"{cl:.10f}")
#print(f"{cdl:.10f}")
print("\n" + "="*60)

# =============================================
# GRÁFICOS COMPARATIVOS
# =============================================

plt.style.use('seaborn-v0_8-notebook')
plt.rcParams.update({'font.size': 12})

# Función para graficar
def plot_data(data):
    D1 = data['D1']
    lambda_odr, error_odr, c_odr, error_c = data['odr_params']
    
    # Ajuste ODR - Línea principal ajuste
    x_fit = np.linspace(min(D1[:,0]), max(D1[:,0]), 100)
    fit = lambda_odr*1e-3 * x_fit + c_odr

    # Ajuste ODR - Región 5-sigma
    nstd = 5. # to draw 5-sigma intervals
    popt_up = lambda_odr + nstd * error_odr
    popt_dw = lambda_odr - nstd * error_odr
    fit_up  = popt_up*1e-3 * x_fit + c_odr + nstd * error_c
    fit_dw  = popt_dw*1e-3 * x_fit + c_odr - nstd * error_c

    plt.errorbar(D1[:,0], D1[:,1], xerr=D1[:,2], yerr=D1[:,3], ecolor='k', fmt='none', label='Datos')
    plt.xlabel('N')
    plt.ylabel('$2 \cdot d$ [μm]')

    plt.plot(x_fit, fit, 'r', lw=2, label='Ajuste lineal \n $\lambda:%1.1f \pm %1.1f\;nm$'%(lambda_odr,error_odr))
    plt.fill_between(x_fit, fit_up, fit_dw, alpha=.25, label='Intervalo 5-sigma')
    plt.legend(loc=4)
    plt.grid(ls='--',color='grey',lw=.5)
    plt.ylim(0,140)
    plt.xlim(0,220)
    plt.title(f'Interferómetro {data["nombre"]}')#, pad=20)
    plt.tight_layout()  # Mejor ajuste de los elementos

    plt.show()

# Generar gráficos
plot_data(mi_data)
plot_data(fp_data)