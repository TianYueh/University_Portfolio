v = [1 4 0 1 1 3];
od = v;
ev = (v(1:end-1) + v(2:end)) / 2;
ans = zeros(1, 2*length(v) - 1);
ans(1:2:end) = od;
ans(2:2:end) = ev