now=0;
x1=0;
x2=0.980506;
h=pi/180000;
big=0;
for i=1:180000
    
    x1new=x1+h*x2;
    x2new=x2-h*(1/4)*x1;
    now=now+h;
    truevalue=2*sin(now/2);
    error=(truevalue-x1new)/(truevalue);
    if(big<abs(error))
        big=abs(error);
    end
    x1=x1new;
    x2=x2new;
end
big