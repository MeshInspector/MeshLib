# the scipt to derive checks within orient3d() function
from sympy import *
ax, bx, cx, ay, by, cy, az, bz, cz, e = symbols('ax bx cx ay by cy az bz cz e')
pax=ax+e**4
pay=ay+e**2
paz=az+e
pbx=bx+e**32
pby=by+e**16
pbz=bz+e**8
pcx=cx+e**256
pcy=cy+e**128
pcz=cz+e**64
d=det(Matrix([[pax,pay,paz],[pbx,pby,pbz],[pcx,pcy,pcz]]))
for i in range(1000):
     c = d.coeff(e,i)
     if c != 0:
         print(f'{i}: {c}')
     if c == 1 or c == -1:
         break
