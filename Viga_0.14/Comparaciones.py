# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Programa para comparar resultados.

Por: Alejandro Hincapie G.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from paper_2d import despl_2d, plot_despl
from leer_GMSH import LaG_from_msh, xnod_from_msh

filename = 'viga'

# %% constantes que ayudarán en la lectura del código
X, Y = 0, 1
NL1, NL2, NL3, NL4, NL5, NL6, NL7, NL8, NL9, NL10 = range(10)

# %% Se lee el archivo de Excel
df       = pd.read_excel(f'{filename}.xlsx', sheet_name=None)

# %% se lee la posicion de los nodos
T = df['xnod'][['nodo', 'x']].set_index('nodo')
xnod    = T['x'].sort_index().to_numpy() # posicion de los nodos
nno  = xnod.size                   # numero de nodos
nef  = nno - 1                     # numero de elementos finitos (EF)


# %% se leen la matriz de conectividad (LaG), el modulo de elasticidad, las
# propiedades del material y las cargas
T     = (df['LaG_EI_q'][['EF', 'NL1', 'NL2', 'E', 'I', 'q1e', 'q2e', 'b', 'h', 'nu']]
         .fillna(0).set_index('EF').sort_index())
LaG   = T[['NL1','NL2']].to_numpy() - 1   # Relación de cada EF con sus nodos
I = T['I'].to_numpy()[0]
q = -T['q1e'].to_numpy()[0]
A = (T['b']*T['h']).iloc[0]
h = T['h'].iloc[0]
E = T['E'].iloc[0]
nu = T['nu'].iloc[0]

# %% Datos necesarios para desplazamientos del paper:

gdl  = np.reshape(np.arange(2*nno), (nno, 2))
idx = [None]*nef      # grados de libertad del elemento e
L = np.diff(xnod)

for e in range(nef):  # ciclo sobre todos los elementos finitos
    idx[e] = np.r_[gdl[LaG[e, 0]], gdl[LaG[e, 1]]]

# Se cargan los datos necesarios:
LaG_p   = T[['NL1','NL2']].to_numpy() - 1   # Relación de cada EF con sus nodos
E_p     = T['E'].to_numpy()                 # modulo de elasticidad E del EF
I_p     = T['I'].to_numpy()                 # momento de inercia Iz del EF
qq_p    = T[['q1e', 'q2e']].to_numpy()      # relación de las cargas distribuidas
bb_p    = T['h'].to_numpy()
hh_p    = T['h'].to_numpy()
nuu_p   = T['nu'].to_numpy()


# %% Se leen los resultados del paper:

despl_p = np.loadtxt('./resultados_paper/despl_paper.csv')
mom_p = np.loadtxt('./resultados_paper/M_paper.csv')
cor_p = np.loadtxt('./resultados_paper/Q_paper.csv')
nor_p = np.loadtxt('./resultados_paper/N_paper.csv')
a = np.loadtxt('./resultados_paper/2d/a_paper.csv')
a_r = np.loadtxt('./resultados_paper/2d/ar_paper.csv')


# %% Resutados EB:

despl_eb = np.loadtxt('./resultados_EB/despl_EB.csv')
mom_eb = np.loadtxt('./resultados_EB/M_EB.csv')
cor_eb = np.loadtxt('./resultados_EB/Q_EB.csv')


# %% Resultados Timo:

despl_t = np.loadtxt('./resultados_Timo/despl_Timo.csv')
mom_t = np.loadtxt('./resultados_Timo/M_Timo.csv')
cor_t = np.loadtxt('./resultados_Timo/Q_Timo.csv')


# %% Resultados de tensión plana:

x_tp = np.loadtxt('./resultados_TP/x_tp.csv')
despl_tp = np.loadtxt('./resultados_TP/w_tp.csv')
res_d3 = pd.read_csv('./resultados_TP/res_d3.csv')
res_esf3 = pd.read_csv('./resultados_TP/res_esf3.csv')

y = res_d3['y'] - h/2

# Esfuerzos y desplazamientos tensión plana en x=x0
ux_tp = res_d3.ux - res_d3.ux.median()
sx_tp = res_esf3.sx - res_esf3.sx.median()


# %% Deformada de la sección:

x0 = 2.5  # x de la sección analizada
n = int(10*x0)  # Nodo ubicado en x=x0

# Timoshenko:

theta_3t = despl_t[n,1]     # Theta en x=x0
ux_t = -y*np.tan(theta_3t)  # u en x=x0
uy_t = despl_t[n,0]         # w en x=x0

# Euler-Bernoulli:

theta_3eb = despl_eb[n,1]
ux_eb = -y*np.tan(theta_3eb)
uy_eb = despl_eb[n,0]

# Paper:

yp, U1, U2 = despl_2d(n, LaG_p, qq_p, L, nuu_p, E_p, hh_p, I_p, idx, a, a_r)
ux_p = U1[:,0] - np.median(U1[:,0])


# %% Esfuerzos sx en la sección:

# Timoshenko:
M_3t = mom_t[n]
sx_t = -M_3t*y/I

# Euler-Bernoulli:
M_3eb = mom_eb[n,0]
sx_eb = -M_3eb*y/I

# Paper:
# =============================================================================
# Se analizarán los esfuerzos en el nodo ubicado en x=2.5, tomándolo como el nodo
# 2 del EF 24 y como el nodo 1 del EF 25, esto darán esfuerzos distintos que al
# final, tal como se hace usualmente con el MEF, se promediarán en el nodo.
# =============================================================================

Le = 0.1
Phi = 3*h**2*(1+nu)/Le**2

# Desplazamientos nodales del nodo 1 del EF 25:
v1, t1, v2, t2 = a[idx[n]]
u1, u2 = a_r[LaG[n]]
u = np.array([u1, v1, t1, u2, v2, t2])
x1 = -Le/2

# Desplazamientos nodales del nodo 2 del EF 24:
v12, t12, v22, t22 = a[idx[n-1]]
u12, u22 = a_r[LaG[n-1]]
u2 = np.array([u12, v12, t12, u22, v22, t22])
x2 = Le/2

# Campo de esfuerzos (deducido del artículo):
def sx(x, y, Phi, u):
    B_sig = (E/Le)*np.array([-1,
                             -12*x*y/(Le**2*(9*Phi + 1)),
                             y*(9*Le*Phi + Le - 6*x)/(Le*(9*Phi + 1)),
                             1,
                             12*x*y/(Le**2*(9*Phi + 1)),
                             -y*(9*Le*Phi + Le + 6*x)/(Le*(9*Phi + 1))])
    p_sig = (-Le**2*q*y - h**3*nu*q - 3*h**2*nu*q*y + 12*q*x**2*y
             - 8*q*y**3)/(24*I)
    
    return B_sig@u + p_sig

# Se calculan los esfuerzos por ambos caminos:
sx_p1 = np.zeros_like(y)
sx_p2 = np.zeros_like(y)
for i in range(y.size):
    sx_p1[i] = sx(x1, y[i], Phi, u)
    sx_p2[i] = sx(x2, y[i], Phi, u2)

# Finalmente se promedian:
sx_p = np.c_[sx_p1, sx_p2].mean(axis=1)



# %% Se realizan los gráficos comparativos:

h_L = h/5  # Relación h/L de la viga


# Desplazamiento y giro:
fig1, ((ax1, ax3), (ax2, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

ax1.plot(xnod, despl_eb[:,0]*1000, 'b-.', label='Euler-Bernoulli')
ax1.plot(xnod, despl_t[:,0]*1000, 'g--', label='Timoshenko')
ax1.plot(xnod, despl_p[:,0]*1000, 'r-', label='EF paper')
ax1.plot(x_tp, despl_tp*1000, 'm.-', label='EF 2D Tensión plana')
ax1.set_title('Desplazamiento vertical')
ax1.set_ylabel('w(x) [mm]')

ax2.plot(xnod, despl_eb[:,1], 'b-.', label='Euler-Bernoulli')
ax2.plot(xnod, despl_t[:,1], 'g--', label='Timoshenko')
ax2.plot(xnod, despl_p[:,1], 'r-', label='EF paper')
ax2.set_title('Ángulo de giro [rad]')
ax2.set_xlabel('x [m]')

fig1.suptitle(f'Comparación de resultados\nh/L = {h_L:.2f}', fontsize='x-large', fontweight='demi')

# Momento y cortante:
#fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(8, 12))

for e in range(nef):
    ax3.plot(xnod[LaG[e]], mom_eb[e], 'b-.', label='Euler-Bernoulli' if e==0 else None)
    ax3.plot(xnod[LaG[e]], [mom_t[e], mom_t[e]], 'g-', label='Timoshenko' if e==0 else None)
    ax3.plot(xnod[LaG[e]], mom_p[e,:], 'r-', label='EF paper' if e==0 else None)

    ax4.plot(xnod[LaG[e]], [cor_eb[e], cor_eb[e]], 'b-.', label='Euler-Bernoulli' if e==0 else None)
    ax4.plot(xnod[LaG[e]], [cor_t[e], cor_t[e]], 'g-', label='Timoshenko' if e==0 else None)
    ax4.plot(xnod[LaG[e]], [cor_p[e], cor_p[e]], 'r-', label='EF paper' if e==0 else None)

ax3.set_title('Momento flector')
ax3.set_ylabel('M(x) [kN-m]')

ax4.set_xlabel('x [m]')
ax4.set_ylabel('Q(x) [kN]')
ax4.set_title('Fuerza cortante')

for ax in (ax1, ax2, ax3, ax4):
    ax.axhline(color='k', lw=1)
    ax.set_xlim(np.amin(xnod), np.amax(xnod))
    ax.legend()

fig1.savefig('./Comparacion/w_t.png')

    
# %% Se grafican resultados en la sección x=x0:

fig3, (ax5, ax6) = plt.subplots(2, 1, figsize=(8, 12))


ax5.plot(ux_eb, y, 'b-.', label='Euler-Bernoulli')
ax5.plot(ux_t, y, 'g--', label='Timoshenko')
ax5.plot(ux_tp, y, 'm.-', label='EF 2D Tensión plana')
ax5.plot(ux_p, yp, 'r-', label='EF Paper')
ax5.set_title(f'Desplazamiento horizontal en x={x0}')
ax5.set_xlabel('u(x) [m]')
ax5.set_ylabel('y [m]')
ax5.set_ylim(y.min(), y.max())
ax5.set_ylim(y.min(), y.max())
ax5.set_yticks(np.linspace(-h/2, h/2, 5))
ax5.legend()

ax6.plot(sx_eb, y, 'b-.', label='Euler-Bernoulli')
ax6.plot(sx_t, y, 'g--', label='Timoshenko')
ax6.plot(sx_tp, y, 'm.-', label='EF 2D Tensión plana')
ax6.plot(sx_p, y, 'r-', label='EF Paper')
ax6.set_title(fr'Esfuerzo $\sigma_x$ en x={x0}')
ax6.set_xlabel('$\sigma_x$ [kPa]')
ax6.set_ylabel('y [m]')
ax6.set_ylim(y.min(), y.max())
ax6.set_yticks(np.linspace(-h/2, h/2, 5))
ax6.legend()

fig3.suptitle(f'Comparación en x = {x0} m\nh/L = {h_L:.2f}', fontsize='x-large', fontweight='demi')
fig3.tight_layout(pad=6.0)
fig3.subplots_adjust(top=0.9)
fig3.savefig('./Comparacion/seccion.png')


# %% Se grafican los desplazamientos 2D del paper:

# Se grafica deformada 2D del paper:
fig4, (ax7, ax8) = plt.subplots(2, 1, figsize=(19, 9))
escala = 100
for e in range(nef):
    plot_despl(e, LaG_p, xnod, qq_p, L, nuu_p, E_p, hh_p, I_p, idx, a, a_r, ax7, escala)
for x in [0, 5]:
    ax7.plot([x, x], [-h/2, h/2], lw=1, c='b')
ax7.set_aspect('equal', adjustable='box')
ax7.set_title(f'Deformada escalada {escala} veces (Paper)')
ax7.set_yticks(np.linspace(-h/2, h/2, 5))
ax7.set_xlabel('x [m]')
ax7.set_ylabel('y [m]')

# Datos de desplazamiento de tensión plana:

# Se carga la malla 2D de tensión plana:

malla = './malla/malla.msh'  # Archivo exportado por GMSH

xnod_tp = xnod_from_msh(malla)  # Se halla la matriz con la función respectiva
nno_tp  = xnod_tp.shape[0]      # número de nodos (número de filas de la matriz xnod)

ngdl = 2*nno_tp            # número de grados de libertad (dos por nodo)
gdl  = np.reshape(np.arange(ngdl), (nno_tp,2)) # nodos vs grados de libertad

LaG = LaG_from_msh(malla)
nef = LaG.shape[0]      # número de EFs (número de filas de la matriz LaG)

atp = np.loadtxt('./resultados_TP/a_tp.csv')  # Vector de desplazamientos TP
delta  = np.reshape(atp, (nno_tp,2))
xdef   = xnod_tp + escala*delta    # posición de la deformada
plt.sca(ax8)
for e in range(nef):
   nod_ef = LaG[e, [NL1, NL4, NL5, NL2, NL6, NL7, NL3, NL8, NL9, NL1]]
   plt.plot(xnod_tp[nod_ef, X], xnod_tp[nod_ef, Y], 'r--',
                        label='Posición original'  if e == 0 else "", lw=0.5)
   plt.plot(xdef[nod_ef, X], xdef[nod_ef, Y], 'b',
                        label='Posición deformada' if e == 0 else "", lw=1)
ax8.set_aspect('equal', adjustable='box')
plt.xlabel('$x$ [m]')
plt.ylabel('$y$ [m]')
plt.title(f'Deformada escalada {escala} veces (Tensión plana)')
ax8.set_yticks(np.linspace(0, h, 5))
fig4.tight_layout()
fig4.subplots_adjust(top=0.9)
fig4.suptitle(f'Campo de desplazamientos 2D\nh/L = {h_L:.2f}', fontsize='x-large', fontweight='demi')
fig4.savefig('./Comparacion/2D.png')


# %% Comparación de desplazamientos verticales 2D en 3 puntos:
# Punto 1 = (2.5,  h/2)
# Punto 2 = (2.5, -h/2)
# Punto 3 = (3, h/2)

# Para desplazamientos del paper

ef_punto = [25, 25, 40]     # EF (viga) donde se encuentra cada punto
y_punto = [h/2, -h/2, h/2]  # Coord. Y de cada punto
Uy_p = np.empty(3)

for i in range(3):
    U1_n = despl_2d(ef_punto[i], LaG_p, qq_p, L, nuu_p, E_p, hh_p, I_p, idx, a, a_r)[1]
    Uy_p[i] = U1_n[:,1][yp == y_punto[i]][0]


# Para desplazamientos de tensión plana:

nodos_tp = [8, 7, 4]     # Nodo de la malla donde se encuentra cada punto

Uy_tp = np.empty(3)

for i in range(3):
    Uy_tp[i] = delta[nodos_tp[i], 1]

# %% Reporte de porcentajes de error 2d:

df = pd.DataFrame(columns=['Punto', 'Uy (Tensión plana) [mm]', 'Uy (Paper) [mm]'])
df['Punto'] = np.arange(1 ,4)
df['Uy (Tensión plana) [mm]'] = Uy_p*1000
df['Uy (Paper) [mm]'] = Uy_tp*1000
df['Error Uy paper [%]'] = ((df['Uy (Tensión plana) [mm]']-df['Uy (Paper) [mm]'])/
                             df['Uy (Tensión plana) [mm]']).abs()*100

df.to_csv('./Comparacion/error_2d.csv', index=False)

# %% Errores en desplazamientos:
# Estos errores se evaluarán en 2 puntos, estos son: x=2.5 y x=4
nodos = np.arange(nno)[np.isin(xnod, [2.5])]
nodos_tp = np.where(np.isclose(x_tp, 2.5))[0][0]

    
df2 = pd.DataFrame(columns=['x', 'uy (Tensión plana) [mm]', 'uy (Paper) [mm]',
                                'uy (Timoshenko) [mm]','Error Paper [%]', 'Error Timoshenko [%]'])

df2.iloc[:,0] = xnod[nodos]                # x a evaluar
df2.iloc[:,1] = despl_tp[nodos_tp]*1000    # Uy tensión plana en eje neutro
df2.iloc[:,2] = despl_p[:, 0][nodos]*1000  # Uy paper en eje neutro
df2.iloc[:,3] = despl_t[:, 0][nodos]*1000  # Uy Timoshenko
df2.iloc[:,4] = ((df2.iloc[:,1] - df2.iloc[:,2])/df2.iloc[:,1]).abs()*100
df2.iloc[:,5] = ((df2.iloc[:,1] - df2.iloc[:,3])/df2.iloc[:,1]).abs()*100

df2.to_csv('./Comparacion/error_desp.csv', index=False)