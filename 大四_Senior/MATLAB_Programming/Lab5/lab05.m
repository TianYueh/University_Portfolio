clear;
TASK_NUM = 8;
test_result = zeros(1, TASK_NUM);

% Test 0: Basic Test
try
    X01 = randn2d(1000);
    X02 = randn2d(1000, [2, 1], 45, [3, 3]);
    X03 = randn2d(1000, [2, 0.5; 0.5, 1], [1, 2]);
    X04 = randn2d(1000, [1, 2], 30, [0, 0], 'plot');
    [X05, Ct05, ut05] = randn2d(1000, [1, 2], 30, [0, 0], 'plot');
    fprintf("Basic Test Pass. \n")
catch ME
    fprintf("Basic Test Fail. \n")
    return
end

% Test 1: error handling - Expected n to be positive
try
    X1 = randn2d(-1000, 'plot');
    fprintf("Test 1 Fail. \n"); % Expected error message
catch ME
    if strcmp(ME.identifier, 'MATLAB:randn2d:expectedPositive') || strcmp(ME.identifier, 'MATLAB:expectedPositive')
        fprintf("Test 1 Pass. \n")
        test_result(1) = true;
    else
        fprintf("Test 1 Fail. \n")
        fprintf("Expected error identifier: %s \n", 'MATLAB:randn2d:expectedPositive');
        fprintf("Your error identifier: %s \n", ME.identifier);
        fprintf("Your error message: %s \n", ME.message);
    end
end

% Test 2: error handling - Expected n to be integer-valued.
try
    X2 = randn2d(140.113, 'plot');
    fprintf("Test 2 Fail. \n") % Expected error message
catch ME
    if strcmp(ME.identifier, 'MATLAB:randn2d:expectedInteger') || strcmp(ME.identifier, 'MATLAB:expectedInteger')
        fprintf("Test 2 Pass. \n")
        test_result(2) = true;
    else
        fprintf("Test 2 Fail. \n")
        fprintf("Expected error identifier: %s \n", 'MATLAB:randn2d:expectedInteger');
        fprintf("Your error identifier: %s \n", ME.identifier);
        fprintf("Your error message: %s \n", ME.message);
    end
end

% Test 3: error handling - Invalid input parameter combination.
try
    X3 = randn2d(1000, 'plot', 'plot');
    fprintf("Test 3 Fail. \n")
catch ME
    fprintf("Test 3 Pass. \n")
    test_result(3) = true;
end

% Test 4: error handling - Invalid input parameter combination.
try
    X4 = randn2d(1, 2, 3, 4, 5, 6);
    fprintf("Test 4 Fail. \n")
catch ME
    fprintf("Test 4 Pass. \n")
    test_result(4) = true;
end

%{
    You must use validateattributes to check the size of C.
%}
% Test 5: error handling - incorrect size of C
try
    X5 = randn2d(1, [1, 2], [3, 4]); % randn2d(n, C, u)
    fprintf("Test 5 Fail. \n")
catch ME
    if strcmp(ME.identifier, 'MATLAB:randn2d:incorrectSize') || strcmp(ME.identifier, 'MATLAB:incorrectSize')
        fprintf("Test 5 Pass. \n")
        test_result(5) = true;
    else
        fprintf("Test 5 Fail. \n")
        fprintf("Expected error identifier: %s \n", 'MATLAB:randn2d:incorrectSize');
        fprintf("Your error identifier: %s \n", ME.identifier);
        fprintf("Your error message: %s \n", ME.message);
    end
end

% Test 6: error handling - Expected s to be positive
try
    X6 = randn2d(1000, [-1, -2], 30, [1, 2]); % randn2d(n, s, a, u)
    fprintf("Test 6 Fail. \n"); % Expected error message
catch ME
    if strcmp(ME.identifier, 'MATLAB:randn2d:expectedPositive') || strcmp(ME.identifier, 'MATLAB:expectedPositive')
        fprintf("Test 6 Pass. \n")
        test_result(6) = true;
    else
        fprintf("Test 6 Fail. \n")
        fprintf("Expected error identifier: %s \n", 'MATLAB:randn2d:expectedPositive');
        fprintf("Your error identifier: %s \n", ME.identifier);
        fprintf("Your error message: %s \n", ME.message);
    end
end

% Test 7: error handling - C is not symmetric
try
    X7 = randn2d(1000, [2, -0.5; 0.5, 1], [1, 2]);
    fprintf("Test 7 Fail. \n"); % Expected error message
catch ME
    fprintf("Test 7 Pass. \n")
    test_result(7) = true;
end

% Test 8: error handling - eigenvalues of C should be positive
try
    X8 = randn2d(1000, [-1, 0; 0, 1], [1, 2]);
    fprintf("Test 8 Fail. \n"); % Expected error message
catch ME
    fprintf("Test 8 Pass. \n")
    test_result(8) = true;
end


if sum(test_result) == TASK_NUM
    fprintf("Congratulations! You passed all the tests. \n")
else
    fprintf("Sorry, you failed some tests. \n")
end