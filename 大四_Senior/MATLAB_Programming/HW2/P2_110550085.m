%110550085房天越

N = 100; 
points = randn(N, 2); % mean = 0, std = 1

theta = 20 * pi / 180; % rotation agle (rad)
scaling = [120, 50]; % scaling factors
translation = [400, 300]; % translation vector

% rot matrix
R = [cos(theta), -sin(theta); sin(theta), cos(theta)];

% apply rotation, scaling, and translation
transformed_points = (R * (points .* scaling)')' + translation;

mu = mean(transformed_points); % mean vector
cov_matrix = cov(transformed_points); % cov vector

inv_cov = inv(cov_matrix); % take the inv of cov

% create meshgrid for the given area x 800, y 600, each axis 200 points
[xgrid, ygrid] = meshgrid(linspace(0, 800, 100), linspace(0, 600, 100));

% combine into grid points
grid_points = [xgrid(:), ygrid(:)];

% diff = each point - mean vector
diff = grid_points - mu;

% the square of mahal distance which is needed later
mahal_dist_sq = sum((diff * inv_cov) .* diff, 2);

% compute p(X) with the given formula
px = exp(-0.5 * mahal_dist_sq) / (2 * pi * sqrt(det(cov_matrix)));
% reshape it to 100*100
px_grid = reshape(px, size(xgrid));

figure;
hold on;

% normalize px to 0~2
px_grid_normalized = (px_grid - min(px_grid(:))) * 2/ (max(px_grid(:)) - min(px_grid(:)));

% display normalized p(x) with imagesc
imagesc(linspace(0, 800, 100), linspace(0, 600, 100), px_grid_normalized);

% apply colormap "summer"
colormap('summer');

% ensure y is upward
set(gca, 'YDir', 'normal'); 

% contour plot mahal dist with labeled contours
contour_levels = [0.5, 1, 2];

% contours parameter
contours = sqrt(mahal_dist_sq);

% contour drawing
[C, h] = contour(xgrid, ygrid, reshape(contours, size(xgrid)), contour_levels, 'k', 'LineWidth', 1); 

% labeling the contour line
clabel(C, h, 'FontSize', 10, 'Color', 'black');

% plot sample points with blue and mean with red cross
plot(transformed_points(:,1), transformed_points(:,2), 'b.', 'MarkerSize', 6);
plot(mu(1), mu(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2); 

% set axis intervals to be 100
xticks(0:100:800); 
yticks(0:100:600);  


axis equal;
hold off;

