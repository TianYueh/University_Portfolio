v = [3 7 5 2];
g1 = v(1)*v(4)+v(3)*v(2);
g2 = v(4)*v(2);
g = gcd(g1, g2);
%fprintf('%d\n', g);
fprintf('%d/%d+%d/%d=%d/%d', v(1), v(2), v(3), v(4), g1, g2)