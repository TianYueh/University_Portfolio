x=[-1, -0.5, 0, 0.5, 1];
f=[0,0,1,0,0];
cubspline1(x, f)

function rtn=cubspline1(x,f)

h=zeros(1, length(x)-1);

for i=1:length(h)
    h(i)=x(i+1)-x(i)
end

A=sparse(length(h)-1, length(h)-1);

for i=1:length(h)-1
    A(i, i)=2*(h(i)+h(i+1));
end
for i=1:length(h)-2
    A(i, i+1)=h(i+1);
end
for i=1:length(h)-2
    A(i+1, i)=h(i+1);
end

y=zeros(length(h)-1, 1);
for i=1:length(h)-1
    y(i)=6*((f(i+2)-f(i+1))/h(i+1)-(f(i+1)-f(i))/h(i));
end


S=A^-1*y;

S=[0;S;0];
a=zeros(length(h),1);
b=zeros(length(h),1);
c=zeros(length(h),1);
d=zeros(length(h),1);
for i=1:length(h)
    a(i)=(S(i+1)-S(i))/(6*h(i));
    b(i)=S(i)/2;
    c(i)=(f(i+1)-f(i))/h(i)-(2*h(i)*S(i)+h(i)*S(i+1))/6;
    d(i)=f(i);
end
polysplinecoeff=[a b c d]
end







