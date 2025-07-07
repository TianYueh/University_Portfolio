A=[4.63, -1.21, 3.22;
    -3.07, 5.48, 2.11;
    1.26, 3.11, 4.57]
b=[2.22;
    -3.17;
    5.11]
x=[0;
    0;
    0]

Gausi(A, b, x,1.46, 0.00001)

function rtn= Gausi(A,b,x,w,tol)
sz=size(A);
n=sz(1);
D=eye(n);
for i=1:n
    D(i,i)=A(i,i);
end
L=zeros(n);
for i=1:n
    for j=1:i
        if(i>j)
            L(i,j)=A(i,j);
        end
    end
end
U=A-D-L;
wLDsuminv=(w*L+D)^-1;
wb=w*b;
wLDsuminvwb=(w*L+D)^-1*wb;
wUDsum=w*(U+D);
iter=0;
wb=w*b
q=0;
while(q==0)
    y=-wLDsuminv*(wUDsum-D)*x+wLDsuminvwb;
    iter=iter+1;
    if(norm(y-x)>tol)
        x=y;
    else
        q=1;
    end
end
x
iter
end
