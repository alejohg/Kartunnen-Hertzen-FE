# -*- coding: utf-8 -*-
"""
Verificación de las deducciones del artículo:
'Exact theory for a linearly elastic interior beam' de Karttunen & Von Hertzen'

Por: Alejandro Hincapié Giraldo
"""

import sympy as sp
import numpy as np
from numpy import sqrt


x, y, z, t, h, q, I, c1, c2, c3, A, N, M, Q, L = sp.symbols('x y z t h q I c1 '
                                                            'c2 c3 A N M Q L')

sp.init_printing()

def my_print(var, nombre):
    print('\n', 80*'~')
    if nombre != '':
        print(f'{nombre} =\n')
    else:
        print()
    sp.pprint(var)

    
# %% Función de tensión de Airy

psi = (c1*y**2 + c2*y**3 + c3*x*y*(1-4*y**2/(3*h**2))
       - q/(240*I)*(5*h**3*x**2 + 15*h**2*x**2*y + 4*y**3*(y**2-5*x**2)))


# %% Esfuerzos:

sx = sp.diff(psi, y, 2)
my_print(sx, 'sx')

sy = sp.diff(psi, x, 2)
my_print(sy, 'sy')

txy = sp.simplify(-sp.diff(psi, x, y))
my_print(txy, 'txy')

my_print('', '')
# %% Fuerzas normal, momento y cortante:

N_x = t*sp.integrate(sx, (y, -h/2, h/2))
my_print(N_x, 'N(x)')


M_x = sp.expand(t*sp.integrate(sx*y, (y, -h/2, h/2)))
my_print(M_x, 'M(x)')

Q_x = sp.expand(t*sp.integrate(txy, (y, -h/2, h/2)))
my_print(Q_x, 'Q(x)')

my_print('', '')


# %% Se verifican las derivadas:

print('\n', 80*'~')
print('Verificación de las ecuaciones:')
print(f'dN/dx = 0: {sp.diff(N_x, x) == 0}')
print(f'dM/dx = Q: {sp.diff(M_x, x) - Q_x == 0}')
print(f'dQ_dx = q: {sp.diff(Q_x.subs(t, 12*I/h**3), x) - q == 0}')

my_print('', '')


# %% Se hallan las constantes c1, c2 y c3:

eq1 = N - N_x  # N = 2*A*c1
eq2 = Q - Q_x  # Q = q*x - 2/3*A*c3
eq3 = M - M_x  # M = 6*I*c2 - 2/3*A*c3*x + q/2*(x^2 - h^2/10)

sol = sp.solve([eq1, eq2, eq3], [c1, c2, c3])


# %% Se reemplazan los valores de c1, c2 y c3 en las fórmulas de sx y sy

Sx = sp.expand(sx.subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3])]))
my_print(Sx, 'Sx')

Txy = sp.expand(txy.subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3])]))
my_print(Txy, 'Txy')

my_print('', '')


# %% Se definen nuevas variables simbólicas:

E, G, nu, C, C1, C2, D1, D2 = sp.symbols('E G nu C C1 C2 D1 D2')
f = sp.Function('f')(y)
g = sp.Function('g')(x)

# %% Se definen las deformaciones para tensión plana:

G = E/(2*(1+nu))

ex = (sx - nu*sy)/E
ey = (sy - nu*sx)/E
gxy = txy/G


# %% Se integran las deformaciones para obtener los desplazamientos:

U_x = sp.integrate(ex, x) + f
U_y = sp.integrate(ey, y) + g

gxy_ = sp.diff(U_x, y) + sp.diff(U_y, x)

# %% 

tmp = (gxy - gxy_)#.expand().collect([x, y])
#tmp2 = tmp + sp.diff(g, x) + sp.diff(f, y)

# %% Para encontrar f(y) derivamos tmp respecto a y

dtmp_dy = tmp.diff(y)

sol = sp.solve(dtmp_dy, f.diff(y, 2))

d2f_dy2 = sol[0]

df_dy = sp.integrate(d2f_dy2, y) + C1

f_y = sp.integrate(df_dy, y) + D1

# %% Para encontrar g(x) se deriva tmp respecto a x

dtmp_dx = tmp.diff(x)

sol = sp.solve(dtmp_dx, g.diff(x, 2))

d2g_dx2 = sol[0]

dg_dx = sp.integrate(d2g_dx2, x) + C2

g_x = sp.integrate(dg_dx, x) + D2


# %% Se reemplazan las derivadas de f(y) y g(x) en gxy:

tmp2 = tmp.subs([(f.diff(y), df_dy), (g.diff(x), dg_dx)]).simplify()

sol = sp.solve(tmp2, C1)

C_1 = sol[0]

f_y = f_y.subs(C1, C_1)

# %% Se reemplazan las funciones halladas f(y) y g(x) en Ux y Uy:

Ux = U_x.subs(f, f_y)

Uy = U_y.subs(g, g_x)


# %% Se define una constante C = C2*E, que se reemplaza en ambas expresiones:
# Al ser D1 y D2 constantes arbitrarias, se hace D1 = D1/E y D2 = D2/E

Ux = Ux.subs([(C2, C/E), (D1, D1/E), (D2, D2/E)]).expand().collect(c3).collect(q*x)

Uy = Uy.subs([(C2, C/E), (D1, D1/E), (D2, D2/E)]).expand().collect(c2).collect(c3).collect(q)

my_print(Ux, 'U_x')
my_print(Uy, 'U_y')

my_print('', '')

# %% Se calculan los desplazamientos en el eje neutro (y=0):

ux = Ux.subs(y, 0)

uy = Uy.subs(y, 0)

phi = Ux.diff(y).subs(y, 0)

# %% Del artículo se tienen las ecuaciones de los desplazamientos Ux, Uy en
# términos de los despl. del eje central. Se verifican:

Ux_c = ux + y*phi - 4*y**3/(3*h**2)*(phi + uy.diff(x)) + nu*y**3/6*phi.diff(x, 2)

Uy_c = (uy - nu*y*ux.diff(x) - nu*y**2/2*phi.diff(x)
        + q*y/(48*E*I)*((2*h+3*y)*(nu**2-1)*h**2 + 2*y**3*(1+2*nu)))

# Se reporta el resultado:
print('\n', 80*'~')
print(f'Ux = Ux_c: {(Ux - Ux_c).simplify() == 0}')
print(f'Uy = Uy_c: {(Uy - Uy_c).simplify() == 0}')

my_print('', '')


# %% Se calculan las deformaciones en términos de los despl. del eje central:

# Primero se definen los DEC como funciones de x:
u_x = sp.Function('u_x')(x)
u_y = sp.Function('u_y')(x)
phi_ = sp.Function('phi')(x)


# Se definen nuevamente los desplazamientos en términos de los DEC:
Ux_c = u_x + y*phi_ - 4*y**3/(3*h**2)*(phi_ + u_y.diff(x)) + nu*y**3/6*phi_.diff(x, 2)

Uy_c = (u_y - nu*y*u_x.diff(x) - nu*y**2/2*phi_.diff(x)
        + q*y/(48*E*I)*((2*h+3*y)*(nu**2-1)*h**2 + 2*y**3*(1+2*nu)))

# Se calculan las deformaciones:
ex_c = sp.diff(Ux_c, x).expand().collect(E)
ey_c = sp.diff(Uy_c, y).expand().collect(Q)
gxy_c = (sp.diff(Ux_c, y) + sp.diff(Uy_c, x)).expand().collect(E)


''' Al ver la expresión resultante de ex_c y compararla con la obtenida en el
    artículo, resulta evidente que dos términos coinciden pero hay 3 términos
    que "sobran" mientras en la expresión del artículo sobra solo 1. Sería ló-
    gico suponer que son expresiones equivalentes, pero se debe demostrar, lo
    cuál se hará a continuación.
    
    Igualmente para el primer término de gxy.
'''
# %% Proceso complementario para verificar la expresión "sobrante" de ex_c:

expr = ex_c - y*phi_.diff(x) - u_x.diff(x)  # expr es lo que "sobra" de ex_c

# Se sustituyen las derivadas de phi y uy (simbólicas) por las derivadas de las
# expresiones de phi y uy obtenidas previamente.

expr_ = expr.subs([(u_y.diff(x,2), uy.diff(x,2)), (phi_.diff(x,3),
                   phi.diff(x,3)), (phi_.diff(x), phi.diff(x))]).simplify()

    
my_print(expr_, 'Expresión "sobrante" de ex_c')

# Luego, esta expresión calculada se reemplaza en la fórmula de ex_c:

ex_c = ex_c - expr + expr_
   
my_print('', '')


# %% Se realiza un proceso similar para ajustar gxy_c al result. del artículo:

expr = gxy_c - (1 - 4*y**2/h**2)*(phi_ + u_y.diff(x))  # Expr. "sobrante" de gxy_

expr_ = expr.subs(u_x.diff(x,2), ux.diff(x,2)).simplify()

my_print(expr_, 'Expresión "sobrante" de gxy_c')  # Se comprueba que el valor es 0

# Y se reemplaza en gxy_c:

gxy_c = gxy_c - expr + expr_
   
my_print('', '')


# %% Se reportan resultados de las deformaciones en términos de los DEC:

my_print(ex_c, 'ex_c (recalculada)')
my_print(ey_c, 'ey_c')
my_print(gxy_c, 'gxy_c (recalculada)')

my_print('', '')


# %% Se definen nuevas variables simbólicas

sigma_x, sigma_y, tau_xy, epsilon_x, epsilon_y, gamma_xy = sp.symbols('sigma_x sigma_y tau_xy epsilon_x epsilon_y gamma_xy')

# Se establecen las relaciones simbólicas esfuerzo - deformación:
e_x = (sigma_x - nu*sigma_y)/E
e_y = (sigma_y - nu*sigma_x)/E
g_xy = tau_xy/G

# Y se hallan las expresiones para los esfuerzos:
sol = sp.solve((epsilon_x - e_x, epsilon_y - e_y, gamma_xy - g_xy), (sigma_x, sigma_y, tau_xy))

s_x, s_y, t_xy = sol.values()  # Asigno estos valores a unas variables nuevas

my_print(s_x, 's_x')
my_print(s_y, 's_y')
my_print(t_xy, 't_xy')

my_print('', '')


# %% Reemplazo los valores de las deformaciones en términos de los DEC:

s_x = s_x.subs([(epsilon_x, ex_c), (epsilon_y, ey_c)])
s_y = s_y.subs([(epsilon_x, ex_c), (epsilon_y, ey_c)])
t_xy = t_xy.subs(gamma_xy, gxy_c)

# %% Calculo las fuerzas de equilibrio con las expresiones anteriores:

N_ = t*sp.integrate(s_x, (y, -h/2, h/2)).subs(I, t*h**3/12).expand()
my_print(N_, 'N(x)')

M_ = (t*sp.integrate(s_x*y, (y, -h/2, h/2))).subs(I, t*h**3/12).expand()
my_print(M_, 'M(x)')

Q_ = sp.expand(t*sp.integrate(t_xy, (y, -h/2, h/2))).subs(I, t*h**3/12).simplify()
my_print(Q_, 'Q(x)')

my_print('', '')


# %% Se reemplazan estas expresiones en las eqns de equilibrio:

dN_dx = N_.diff(x).subs(t, A/h)
dM_dx = M_.diff(x).subs(t, 12*I/h**3)
dQ_dx = Q_.diff(x).subs(t, A/h)

my_print(dN_dx, 'dN/dx = 0')
my_print(dM_dx - Q_, 'dM/dx - Q = 0')
my_print(dQ_dx - q, 'dQ/dx - q = 0')


my_print('', '')


# %% Ecuaciones de energía:

# Energía de deformación:

v_s = sp.Matrix([sx, sy, txy])  # Vector de esfuerzos
v_e = sp.Matrix([ex, ey, gxy])  # Vector de deformaciones

U = 1/2*t*sp.integrate(v_s.dot(v_e), (x, -L/2, L/2), (y, -h/2, h/2))
U = U.subs(t, 12*I/h**3).nsimplify().expand()
my_print(U, 'U')

# Trabajo debido a la carga q:
W_q = -sp.integrate(q*Uy.subs(y, h/2), (x, -L/2, L/2))
my_print(W_q, 'W_q')

# Trabajo de los esfuerzos sobre las superficies laterales:
W_s = t*( sp.integrate(sx.subs(x, L/2)*Ux.subs(x, L/2), (y, -h/2, h/2))
         -sp.integrate(sx.subs(x,-L/2)*Ux.subs(x,-L/2), (y, -h/2, h/2))
         +sp.integrate(txy.subs(x, L/2)*Uy.subs(x, L/2), (y, -h/2, h/2))
         -sp.integrate(txy.subs(x,-L/2)*Uy.subs(x,-L/2), (y, -h/2, h/2)))

W_s = W_s.subs(t, 12*I/h**3).nsimplify().expand()
my_print(W_s, 'W_s')

my_print('', '')


# %% Se verifica que: 2U - W_q - Ws = 0

expr = (2*U - W_s - W_q).simplify()

my_print(expr, '2U - W_q - Ws')
my_print('', '')


# %% Formulación de elementos finitos:

# Se definen los desplazamientos nodales ux_i, uy_i, psi_i para i = 1,2

ux_1, ux_2, uy_1, uy_2, phi_1, phi_2 = sp.symbols('ux_1, ux_2, uy_1, uy_2, phi_1, phi_2')

# Se plantean las ecuaciones de los desplazamientos en nodos 1 y 2...
# en términos de los DEC:

ux1 = ux.subs(x, -L/2)
ux2 = ux.subs(x,  L/2)

uy1 = uy.subs(x, -L/2)
uy2 = uy.subs(x,  L/2)

phi1 = -phi.subs(x, -L/2)
phi2 = -phi.subs(x,  L/2)


# %% Se define una constante Phi que servirá más adelante:
# Phi = 3*h^2(1+nu)/L^2

Phi = sp.Symbol('Phi')  # No confundir con 'phi' minúscula, el giro de la sec.


# %% De las eqns anteriores se hallan las constantes c1, c2,c3, C, D1 y D2
eqns = (ux_1 - ux1, ux_2 - ux2, uy_1 - uy1, uy_2 - uy2,
        phi_1 - phi1, phi_2 - phi2)

sol = sp.solve(eqns, (c1, c2, c3, C, D1, D2))

for key in sol:
    print(f'{key} =')
    sp.pprint(sol[key].simplify())
    print(80*'~', '\n')

my_print('', '')

# %% Reemplazando la constante Phi en c3, C y D2:

for key in (c3, C, D2):
    print(f'{key} =')
    sp.pprint(sol[key].subs(nu, (L/h)**2*Phi/3 - 1).simplify())

my_print('', '')


# %% Se definen las 6 fuerzas nodales:

#N1, N2, M1, M2, Q1, Q2 = sp.symbols('N1, N2, M1, M2, Q1, Q2')


# %% Ahora se obtienen las expresiones para la Normal, Momento y Cortante:

N_1 = -N_x.subs(x, -L/2).subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3])])
N_2 =  N_x.subs(x,  L/2).subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3])])

M_1 =  M_x.subs(x, -L/2).subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3])])
M_2 = -M_x.subs(x,  L/2).subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3])])

Q_1 = -Q_x.subs(x, -L/2).subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3])])
Q_2 =  Q_x.subs(x,  L/2).subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3])])

# Se sustituye el área en las ecuaciones de fuerza axial:

N_1 = N_1.subs(t, A/h)
N_2 = N_2.subs(t, A/h)

# Se sustituye la inercia en las ecuaciones de momento y cortante:
M_1 = M_1.subs(t, 12*I/h**3)
M_2 = M_2.subs(t, 12*I/h**3)
Q_1 = Q_1.subs(t, 12*I/h**3)
Q_2 = Q_2.subs(t, 12*I/h**3)


# %% Se deduce la matriz de rigidez Kr y el vector fr para el elemento barra:

Kr = sp.zeros(2)
fr = sp.zeros(2,1)

i = 0  # identificador para filas

for eqn in (N_1, N_2):

    coef_K = sp.poly(eqn, (ux_1, ux_2)).coeffs()[:2]
    coef_f = sp.poly(eqn, (ux_1, ux_2)).coeffs()[-1]
    for j in range(2):  # j = identificador para columnas
        Kr[i, j] = coef_K[j].simplify()
    fr[i] = coef_f.subs([(A, t*h), (I, t*h**3/12)]).simplify()
    i += 1


fc = E*A/L  # Se define el factor común de Kr (según eq. 41 del artículo)

# Se expresa K como el producto del factor común por la matriz factorizada:
K_r = sp.MatMul(fc, Kr/fc)
my_print(K_r, 'Kr (Matriz de rigidez barra)')

fcf = q*nu*h/2  # Se define el factor común de fr (según eq. 41 del artículo)

f_r = sp.MatMul(fcf, fr/fcf)
my_print(f_r, 'fr (Vector de fuerzas barra)')

my_print('', '')


# %% Se deduce la matriz de rigidez Kb y el vector de fuerzas fb
# para el elemento viga:

Kb = sp.zeros(4)
fb = sp.zeros(4,1)

i = 0  # identificador para filas

for eqn in (Q_1, M_1, Q_2, M_2):
    
    eqn = eqn.subs(nu, (L/h)**2*Phi/3 - 1) # Se reemplaza la constante Phi
    coef_K = sp.poly(eqn, (uy_1, phi_1, uy_2, phi_2)).coeffs()[:4]
    coef_f = sp.poly(eqn, (uy_1, phi_1, uy_2, phi_2)).coeffs()[-1]
    for j in range(4):  # j = identificador para columnas
        Kb[i, j] = coef_K[j].simplify()
    fb[i] = coef_f.subs(Phi, 3*h**2*(1+nu)/L**2).simplify()
    
    i += 1

fc = E*I/((1+Phi)*L**3)  # factor común de K (según eq. 42 del artículo)

# Se expresa K como el producto del factor común por la matriz factorizada:
K_b = sp.MatMul(fc, Kb/fc)
my_print(K_b, 'Kb')

# Mismo proceso para fb:

fcf = q/2  # factor común de f (según eq. 42 del artículo)

f_b = sp.MatMul(fcf, fb/fcf)
my_print(f_b, 'fb (Vector de fuerzas viga)')
my_print('', '')


# %% Cálculo de momentos, cortantes y axiales:

Ee, Ae, Le, qe, Ie = sp.symbols('Ee, Ae, Le, qe, Ie')

MM = M_x.subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3])]).subs(t, 12*I/h**3).expand()
QQ = Q_x.subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3])]).subs(t, 12*I/h**3).expand()
NN = N_x.subs(c1, sol[c1]).subs(t, A/h)

M1 = -MM.subs(x, -L/2)
M2 = -MM.subs(x, L/2)

Q0 = -QQ.subs(x, 0)

coef_N =  sp.poly(NN, (ux_1, ux_2)).coeffs()
coef_M1 = sp.poly(M1, (uy_1, phi_1, uy_2, phi_2)).coeffs()
coef_M2 = sp.poly(M2, (uy_1, phi_1, uy_2, phi_2)).coeffs()

coef_Q0 = sp.poly(Q0, (uy_1, phi_1, uy_2, phi_2)).coeffs()


# %% Matriz Bn para obtener la fuerza axial
B_n = sp.zeros(1, 2)

for j in range(2):
    B_n[j] = coef_N[j].simplify().subs([(E, Ee), (L, Le), (q, qe), (A, Ae), (I, Ie)])

fcn = Ae*Ee/Le
B_n = sp.MatMul(fcn, B_n/fcn)

# %% Se deduce la matriz Bb para obtener los momentos
B_b = sp.zeros(2, 4)

for j in range(4):
    B_b[0, j] = coef_M1[j].simplify().subs([(E, Ee), (L, Le), (q, qe), (A, Ae), (I, Ie)])
    B_b[1, j] = coef_M2[j].simplify().subs([(E, Ee), (L, Le), (q, qe), (A, Ae), (I, Ie)])

fc_Bb = Ee*Ie/(Le**2+3*h**2*nu+3*h**2)  # Factor común de la matriz Bb
B_b = sp.MatMul(fc_Bb, B_b/fc_Bb)

my_print(B_b, 'Bb')
my_print('', '')


# %% Se deduce la matriz Bs para obtener las cortantes

B_s = sp.zeros(1, 4)

for j in range(4):
    B_s[j] = coef_Q0[j].simplify().subs([(E, Ee), (L, Le), (q, qe), (A, Ae), (I, Ie)])

fc_Bs = fc_Bb  # Factor común de la matriz Bs
B_s = sp.MatMul(fc_Bs, B_s/fc_Bs)

my_print(B_s, 'Bs')
my_print('', '')


# %% Por último, se pueden reemplazar las constantes c1, c2, c3, C, D1, D2
# en las ecuaciones de Ux y Uy, para obtenerlas en función de los DEC:

Uxx = Ux.subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3]),
               (C, sol[C]), (D1, sol[D1]), (D2, sol[D2])]).subs(t, 12*I/h**3).expand()
Uyy = Uy.subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3]),
               (C, sol[C]), (D1, sol[D1]), (D2, sol[D2])]).subs(t, 12*I/h**3).expand()


# Para expresar estos desplazamientos en forma matricial, es decir:
# Ux = Nx*u + q*L1 y Uy = Ny*u + q*L1
# se deducen las matrices Nx y Ny

Nx = sp.zeros(1, 6)
Ny = sp.zeros(1, 6)

coef_Ux = sp.poly(Uxx, (ux_1, uy_1, phi_1, ux_2, uy_2, phi_2)).coeffs()
coef_Uy = sp.poly(Uyy, (ux_1, uy_1, phi_1, ux_2, uy_2, phi_2)).coeffs()

for i in range(6):
    Nx[i] = coef_Ux[i].subs(h, sp.sqrt(3*Phi*L**2/(nu+1))).simplify()
    Ny[i] = coef_Uy[i].subs(h, sp.sqrt(3*Phi*L**2/(nu+1))).simplify()

L1 = (coef_Ux[-1]/q).simplify()
L2 = (coef_Uy[-1]/q).simplify()


# %% Matriz para hallar el campo de esfuerzos sx:

Sxx = sx.subs([(c1, sol[c1]), (c2, sol[c2]), (c3, sol[c3]),
               (C, sol[C]), (D1, sol[D1]), (D2, sol[D2])]).expand()

B_sig = sp.zeros(1, 6)

coef_sig = sp.poly(Sxx, (ux_1, uy_1, phi_1, ux_2, uy_2, phi_2)).coeffs()
for i in range(6):
    B_sig[i] = coef_sig[i].subs(h, sp.sqrt(3*Phi*L**2/(nu+1))).simplify()
    
L_sig = (coef_sig[-1]/q).simplify()

fc = E/L

B_sig = sp.MatMul(fc, B_sig/fc)

# %% bye, bye!