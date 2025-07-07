format long

y0=0;
z0=1.2;
h=pi/4;

y1=y0+h*z0;
z1=z0+h*(-y0/4);
y2=y1+h*z1;
z2=z1+h*(-y1/4);
y3=y2+h*z2;
z3=z2+h*(-y2/4);
y4=y3+h*z3;
y4