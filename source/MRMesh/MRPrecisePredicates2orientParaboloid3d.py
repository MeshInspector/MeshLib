# the scipt to derive checks within orientParaboloid3d() function
from sympy import *
ax, bx, ay, by, cx, cy, e = symbols('ax bx ay by cx cy e')
pax=ax+e**3  #instead of **2 in the article, we use **3 here to make all products below have unique power of e, this does not change the checks in ccw
pay=ay+e
pbx=bx+e**27
pby=by+e**9
pcx=cx+e**243
pcy=cy+e**81
d=det(Matrix([[pax,pay,pax*pax+pay*pay],[pbx,pby,pbx*pbx+pby*pby],[pcx,pcy,pcx*pcx+pcy*pcy]]))
for i in range(103):
     c = d.coeff(e,i)
     if c != 0:
         print(f'{i}: {c}')
