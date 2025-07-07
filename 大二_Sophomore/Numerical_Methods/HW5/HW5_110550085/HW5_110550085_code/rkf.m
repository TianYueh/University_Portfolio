[t1, y1] = ode45(@vdp, [0, 0.2], [0, 1, 0]');
[t2, y2] = ode45(@vdp, [0, 0.4], [0, 1, 0]');
[t3, y3] = ode45(@vdp, [0, 0.6], [0, 1, 0]');

fprintf('y(0.2) = %f\n',y1(end,1));
fprintf('y(0.4) = %f\n',y2(end,1));
fprintf('y(0.6) = %f\n',y3(end,1));

function dy = vdp(t, y)
dy = [y(2) y(3) y(1)*2-t*y(2)+t]';
end
