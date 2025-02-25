import sympy

j,w=sympy.symbols("j,w")
j=w/1+w**2
dj_dw=sympy.diff(j,w)
print(dj_dw)