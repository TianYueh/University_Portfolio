clear; 
clc; 
close all;

rng(240926)
xi = -9:3:9;
yi = -.04*xi.^2 + .1*xi + 2 + 1*(rand(1,length(xi))-.5);

degrees = [1, 2, 4];

x_fine = -10:.1:10;

figure;

for i = 1:length(degrees)

    r = degrees(i);
    
    % make matrix A
    A = zeros(length(xi), r+1);
    for j = 0:r
        A(:,j+1) = xi'.^j; 
    end
    
    coeff = A\yi'; 
    
    % predict y on the fine grid
    A_fine = zeros(length(x_fine), r+1);
    for j = 0:r
        A_fine(:,j+1) = x_fine'.^j;
    end
    y_fine = A_fine * coeff;
    
    % predict y for the given points
    y_pred = A * coeff;
    
    %rms
    rms_error = sqrt(mean((yi' - y_pred).^2));
    
    % plot results in subplot, row = 1, column = 3
    subplot(1, 3, i);
    hold on;
    
    % plot original pints
    plot(xi, yi, 'ro', 'LineWidth', 1);
    
    % plot the curve
    plot(x_fine, y_fine, 'b');
    
    % plot the connecting line
    for k = 1:length(xi)
        plot([xi(k), xi(k)], [yi(k), y_pred(k)], 'r-');
    end

    
    % add title and rms text
    title(sprintf('Polynomial deg = %d', r));
    text(0, -2.3, sprintf('rms = %.4f', rms_error), 'FontSize', 10);
    
    ylim([-3 3]);

    % set labels
    xlabel('x');
    ylabel('y');
    
    % Hold off for next plot
    hold off;
end



