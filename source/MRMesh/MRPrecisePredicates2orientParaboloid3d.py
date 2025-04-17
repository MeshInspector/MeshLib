# the scipt to derive checks within orientParaboloid3d() function
from sympy import *
ax, bx, ay, by, cx, cy, e = symbols('ax bx ay by cx cy e')
# instead of **2 in the article, we use **3 here to make the coefficients below independent on point indices
# (with **2 the squares of epsilons can have same power as the product of other epsilons, which spoils coefficients),
# this does not change the checks in ccw
pax=ax+e**3
pay=ay+e
pbx=bx+e**27
pby=by+e**9
pcx=cx+e**243
pcy=cy+e**81
d=det(Matrix([[pax,pay,pax*pax+pay*pay],[pbx,pby,pbx*pbx+pby*pby],[pcx,pcy,pcx*pcx+pcy*pcy]]))
for i in range(1000):
     c = d.coeff(e,i)
     if c != 0:
         print(f'{i}: {c}')
     if c == 1 or c == -1:
         break
