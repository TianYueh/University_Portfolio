x = 1:100;
a1 = ones(1, 100);
a = 2;
a1 = a1*a;

an = a1.^x;
acum = cumprod(x);
ans1 = an./acum;
ans2 = cumsum(ans1);
ans = ans2(100) + 1
