% ========== Setting ==========
filePath = 'lab06_input.txt'; 
% =============================

% ========== Test 1 Display ==========
disp('========== Test 1 Display ==========');
my_word_count(filePath, 'word+');

% ========== Test 2 Sort Mode ==========
disp('========== Test 2 Sort Mode ==========');
flag2 = true;
try
    A1 = my_word_count(filePath, 'word+');
catch ME
    disp("[x] Test 2-1 (word+) fail due to:")
    disp(ME.message);
    flag2 = false;
end
try
    A2 = my_word_count(filePath, 'word-');
catch ME    
    disp("[x] Test 2-2 (word-) fail due to:")
    disp(ME.message);
    flag2 = false;
end
try
    A3 = my_word_count(filePath, 'len+');
catch ME
    disp("[x] Test 2-3 (len+) fail due to:")
    disp(ME.message);
    flag2 = false;
end
try
    A4 = my_word_count(filePath, 'len-');
catch ME
    disp("[x] Test 2-4 (len-) fail due to:")
    disp(ME.message);
    flag1 = false;
end
try
    A5 = my_word_count(filePath, 'count+');
catch ME
    disp("[x] Test 2-5 (count+) fail due to:")
    disp(ME.message);
    flag1 = false;
end
try
    A6 = my_word_count(filePath, 'count-');
catch ME
    disp("[x] Test 2-6 (count-) fail due to:")
    disp(ME.message);
    flag1 = false;
end

if flag2
    fprintf("[v] Congratulations! All tests in Test 2 Pass. \n")
else
    fprintf("[x] Sorry! Some tests in Test 2 Fail. \n")
end

% ========== Test 3 Correctness ==========
disp('========== Test 3 Correctness ==========');

golden = load('lab06_golden.mat');
flag3 = true;
flag3 = check(A1, golden.A1, 1) && flag3; % word+
flag3 = check(A2, golden.A2, 2) && flag3; % word-
flag3 = check(A3, golden.A3, 3) && flag3; % len+
flag3 = check(A4, golden.A4, 4) && flag3; % len-
flag3 = check(A5, golden.A5, 5) && flag3; % count+
flag3 = check(A6, golden.A6, 6) && flag3; % count-

if flag3
    fprintf("[v] Congratulations! All tests in Test 3 Pass. \n")
else
    fprintf("[x] Sorry! Some tests in Test 3 Fail. \n")
end

% ========== Test 4 Error Handling ==========
disp('========== Test 4 Error Handling ==========');
flag4 = true;
try
    my_word_count(filePath, 'word');
    fprintf("[x] Test 4-1 Fail. \n")
    flag4 = false;
catch ME

end

try
    my_word_count(filePath, 'word+-');
    fprintf("[x] Test 4-2 Fail. \n")
    flag4 = false;
catch ME

end

try
    my_word_count(filePath, 'size');
    fprintf("[x] Test 4-3 Fail. \n")
    flag4 = false;
catch ME

end

try
    my_word_count(filePath, 'len^');
    fprintf("[x] Test 4-4 Fail. \n")
    flag4 = false;
catch ME

end

try
    my_word_count("not_exist", 'word+');
    fprintf("[x] Test 4-5 Fail. \n")
    flag4 = false;
catch ME

end

if flag4
    fprintf("[v] Congratulations! All tests in Test 4 Pass. \n")
else
    fprintf("[x] Sorry! Some tests in Test 4 Fail. \n")
end

function flag = check(X, X_golden, task_id)
    % Make sure X is a row vector
    [x_row, ~] = size(X);
    if x_row ~= 1
        X = reshape(X, 1, []);
    end
    flag = isequal(X, X_golden);
    if ~flag
        fprintf("[x] Test 3-%d Fail. \n", task_id)
        if isstruct(X) && isstruct(X_golden)
            if length(X) ~= length(X_golden)
                fprintf("Array lengths differ. Yours: %d, Expected: %d\n", length(X), length(X_golden))
            else
                disp("We have not implemented the comparison of struct yet.")
            end
        else
            disp("Your output should be struct, and the field names should be 'word', 'len', 'count'.")
        end
    end
end
