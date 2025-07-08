clc;
clear;

score = 0;
% ========== Test 1 Correctness ==========
disp("========== Test 1 Correctness ==========");

golden_file = 'P3golden.txt';
result_file = 'P3result.txt';

golden_fid = fopen(golden_file, 'r');
result_fid = fopen(result_file, 'r');

if golden_fid == -1 || result_fid == -1
    error('Cannot open file');
end

% Initialize variables
line_idx = 0;
flag = true;

% Read and compare line by line
while ~feof(golden_fid) && ~feof(result_fid)
    line_idx = line_idx + 1;
    golden_line = fgetl(golden_fid);
    result_line = fgetl(result_fid);
    
    % Compare non-empty lines
    if strcmp(golden_line, result_line)
        % Calculate score based on line number
        if line_idx <= 5
            score = score + 2;  % 2 points for first 5 lines
        else
            score = score + 1;  % 1 point for remaining lines
        end
    else
        fprintf('Mismatch at line %d:\n', line_idx);
        fprintf('Golden: %s\n', golden_line);
        fprintf('Result: %s\n', result_line);
        flag = false;
    end
end

if ~feof(golden_fid) || ~feof(result_fid)
    fprintf('Files are of different lengths.\n');
    flag = false;
end

fclose(golden_fid);
fclose(result_fid);

% Final result
if flag
    fprintf('All tests in Test 1 passed!\n');
else
    fprintf('Some tests in Test 1 failed.\n');
end

part1_score = score;

% ========== Test 2 Error Handling ==========
disp("========== Test 2 Error Handling ==========");
n_subtasks = 5;
pass_count = 0;

% 1. `num1` 或 `den1` 或 `num2` 或 `den2` 不是整數，為其他字元：$5\%$
% 2. `num1` 或 `den1` 或 `num2` 或 `den2` 不是整數，為浮點數：$5\%$
% 3. 分母為 $0$：$10\%$
% 4. `operator` 不是 `+`、`-` 或 `*`：$5\%$
%     - 由於 Bonus 期望支援 `^` 或 `%` 運算，因此測試錯誤處理時不會包含可以作為運算子的字元。
% 5. Missing argument：$5\%$

% Test 2-1: Other characters
try
    [num, den] = FR_arith(1, 2, '3', 4, '+'); % '3' 會被解讀成 ASCII 的 '3'，也就是 51，但這是不對的
    if ~isempty(num) || ~isempty(den)
        disp("Test 2-1: Failed (Other characters) (5%)");
    else
        disp("Test 2-1: Passed (Other characters) (5%)");
        pass_count = pass_count + 1;
        score = score + 5;
    end
catch ME % 預期會報錯，且原本應該不會報錯
    disp("Test 2-1: Passed (Other characters) (5%)");
    pass_count = pass_count + 1;
    score = score + 5;
end

% Test 2-2: Floating-point number
try
    [num, den] = FR_arith(1, 2, 3.14, 4, '+');
    if ~isempty(num) || ~isempty(den)
        disp("Test 2-2: Failed (Floating-point number) (5%)");
    else
        disp("Test 2-2: Passed (Floating-point number) (5%)");
        pass_count = pass_count + 1;
        score = score + 5;
    end
catch ME
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'MATLAB:gcd')
        disp("Test 2-2: Passed (Floating-point number) (5%)");
        pass_count = pass_count + 1;
        score = score + 5;
    else
        disp("Test 2-2: Failed (Floating-point number) (5%)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

% Test 2-3: Division by Zero
try
    [num, den] = FR_arith(1, 2, 3, 0, '+');
    if ~isempty(num) || ~isempty(den)
        disp("Test 2-3: Failed (Division by Zero) (10%)");
    else
        disp("Test 2-3: Passed (Division by Zero) (10%)");
        pass_count = pass_count + 1;
        score = score + 10;
    end
catch ME % 預期會報錯，且原本應該不會報錯
    disp("Test 2-3: Passed (Division by Zero) (10%)");
    pass_count = pass_count + 1;
    score = score + 10;
end

% Test 2-4: Invalid operator
try
    [num, den] = FR_arith(1, 2, 3, 4, '$');
    if ~isempty(num) || ~isempty(den)
        disp("Test 2-4: Failed (Invalid operator) (5%)");
    else
        disp("Test 2-4: Passed (Invalid operator) (5%)");
        pass_count = pass_count + 1;
        score = score + 5;
    end
catch ME % 預期會報錯，且原本應該不會報錯
    disp("Test 2-4: Passed (Invalid operator) (5%)");
    pass_count = pass_count + 1;
    score = score + 5;
end

% Test 2-5: Missing argument
try
    [num, den] = FR_arith(1, 2, 3, 4);
    disp("Test 2-5: Failed (Missing argument) (5%)");
catch ME
    if ~strcmp(ME.identifier, 'MATLAB:minrhs')
        disp("Test 2-5: Passed (Missing argument) (5%)");
        pass_count = pass_count + 1;
        score = score + 5;
    else
        disp("Test 2-5: Failed (Missing argument) (5%)");
        fprintf('\tYour function should raise an error by itself, but your error message is from MATLAB.\n');
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

if pass_count == n_subtasks
    fprintf('All tests in Test 2 passed!\n');
else
    fprintf('Some tests in Test 2 failed.\n');
end

disp("========== Total Score ==========");
fprintf('HW3 score: %d/60 (Part 1: %d/30, Part 2: %d/30)\n', score, part1_score, score - part1_score);

