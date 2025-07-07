A=[4 -1 0 0 0 0;
    -1 4 -1 0 0 0;
    0 -1 4 -1 0 0;
    0 0 -1 4 -1 0;
    0 0 0 -1 4 -1;
    0 0 0 0 -1 4;]

b=[100;
    200;
    200;
    200;
    200;
    100]

TriDiagSym(A,b)

function rtn=TriDiagSym(A,y)

C=spdiags(A);
sz=size(A);
n=sz(1);
k=n-1;
a=zeros(1,n);
b=zeros(1,k);
for i=1:n
    a(i)=C(i,2);
    b(i)=C(i,1);
end

for i=2:(n-1)
    b(i+1)=C(i,1);
end

n=length(y);
v=zeros(n,1);
x=v;

w=a(1);
x(1)=y(1)/w;

for i=2:n
    v(i-1)=b(i-1)/w;
    w=a(i)-b(i)*v(i-1);
    x(i)=(y(i)-b(i)*x(i-1))/w;
end

for j=n-1:-1:1
    x(j)=x(j)-v(j)*x(j+1);
end

x
end