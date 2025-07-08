n = 87;
r = 6.3;

if mod(n, 2) == 0
   n = n - 1;
end

[X, Y] = meshgrid(1:n, 1:n);
center = (n+1)/2;
    
D = sqrt((X - center).^2 + (Y - center).^2);

A = zeros(n);
A(D < r) = 1;
    
A

tester = lab2_tester();
tester.test1(A, 87, 6.3);


