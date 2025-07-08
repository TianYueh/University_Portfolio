function [X, Ct, ut] = randn2d(n, varargin)
    % Check the properties of n
    validateattributes(n, {'numeric'}, {'scalar', 'integer', 'positive'});

    % default vals
    s = [1, 1]; % std in x and y
    a = 0;      % rot angle in deg
    u = [0, 0]; % displacement in x and y
    C = eye(2); % default cov
    plotFlag = false;
    customCov = false;

    % Check for 4.
    if ~isempty(varargin) && ischar(varargin{end}) && strcmpi(varargin{end}, 'plot')
        plotFlag = true;
        varargin(end) = []; % remove 'plot'
    end

    % Check remaining cases
    if isempty(varargin) % Case 1
        X = randn(n, 2); 
    elseif length(varargin) == 2 % Case 3
        C = varargin{1};
        u = varargin{2};
        validateattributes(C, {'numeric'}, {'size', [2, 2]});
        validateattributes(u, {'numeric'}, {'size', [1, 2]});
        if ~issymmetric(C) && any(eig(C) <= 0)
            error('Covariance matrix must be symmetric and positive definite.');
        elseif ~issymmetric(C)
            error('Covariance matrix must be symmetric.');
        elseif any(eig(C) <= 0)
            error('Covariance matrix must be positive definite.');
        end
        customCov = true;
    elseif length(varargin) == 3 % Case 2
        s = varargin{1};
        a = varargin{2};
        u = varargin{3};
        validateattributes(s, {'numeric'}, {'size', [1, 2], 'positive'});
        validateattributes(a, {'numeric'}, {'scalar'});
        validateattributes(u, {'numeric'}, {'size', [1, 2]});
    else
        error('Invalid number or type of input arguments.');
    end

    
    if customCov % Case 3
        [V, D] = eig(C);
        X = randn(n, 2) * sqrt(D) * V'; % sqrt(D) is scaling factor, V' is rotation
        X = X + u;
    else %Case 2
        a = deg2rad(a); 
        R = [cos(a), -sin(a); sin(a), cos(a)]; % rot matrix
        S = diag(s); % scaling
        X = randn(n, 2) * S * R';
        X = X + u;
    end

    % Case 5
    if nargout == 3
        ut = mean(X, 1); 
        Ct = cov(X);     % cov matrix
    end

    
    if plotFlag %Case 4
        figure;
        scatter(X(:, 1), X(:, 2), 10, 'filled');
        title('Generated 2D Samples');
        xlabel('X1');
        ylabel('X2');
        axis equal;
        grid on;
    end
end
