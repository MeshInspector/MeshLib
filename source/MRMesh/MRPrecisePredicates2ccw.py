# the scipt to derive checks within ccw() function
from sympy import *
ax, bx, ay, by, e = symbols('ax bx ay by e')
pax=ax+e**2
pay=ay+e
pbx=bx+e**8
pby=by+e**4
d=det(Matrix([[pax,pay],[pbx,pby]]))
for i in range(1000):
     c = d.coeff(e,i)
     if c != 0:
         print(f'{i}: {c}')
     if c == 1 or c == -1:
         break
