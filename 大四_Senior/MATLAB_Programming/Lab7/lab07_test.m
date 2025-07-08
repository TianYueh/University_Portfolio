clc;
clear;

% ========== Test 1 Correctness ==========
n_subtasks = 6;
pass_count = 0;
disp("========== Test 1 Correctness ==========");

% Test 1-1: Constructor & Disp (1,2,3)
try
    v1 = Vec3(1, 2, 3);
    output = evalc('disp(v1)');  % Capture disp output
    output = strtrim(output);
    expected_output = "(1,2,3)";  % Expected output (Notice the newline character)
    if v1.x == 1 && v1.y == 2 && v1.z == 3 && strcmp(output, expected_output)
        disp('Test 1-1: Passed (constructor & disp)');
        pass_count = pass_count + 1;
    elseif v1.x == 1 && v1.y == 2 && v1.z == 3
        fprintf('\tExpected: (1,2,3), Got: %s\n', output);
        fprintf('\tCheck your disp function.\n');
    elseif strcmp(output, expected_output)
        fprintf('\tExpected: (1,2,3), Got: (%d,%d,%d)\n', v1.x, v1.y, v1.z);
        fprintf('\tCheck your properties.\n');
    else
        disp('Test 1-1: Failed (constructor & disp)');
    end
catch ME
    disp('Test 1-1: Failed (constructor & disp)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 1-2: Constructor & Disp (0,0,0)
try
    v2 = Vec3;
    output = evalc('disp(v2)');  % Capture disp output
    output = strtrim(output);  % Remove leading and trailing whitespace
    expected_output = "(0,0,0)";  % Expected output (Notice the newline character)
    if v2.x == 0 && v2.y == 0 && v2.z == 0 && strcmp(output, expected_output)
        disp('Test 1-2: Passed (constructor & disp)');
        pass_count = pass_count + 1;
    elseif v2.x == 0 && v2.y == 0 && v2.z == 0
        fprintf('\tExpected: (0,0,0), Got: %s\n', output);
        fprintf('\tCheck your disp function.\n');
    elseif strcmp(output, expected_output)
        fprintf('\tExpected: (0,0,0), Got: (%d,%d,%d)\n', v2.x, v2.y, v2.z);
        fprintf('\tCheck your properties.\n');
    else
        disp('Test 1-2: Failed (constructor & disp)');
    end
catch ME
    disp('Test 1-2: Failed (constructor & disp)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 1-3: Plus operator overloading (1,2,3) + (4,5,6)
try
    v3 = Vec3(1, 2, 3) + Vec3(4, 5, 6);
    output = evalc('disp(v3)');  % Capture disp output
    output = strtrim(output);
    expected_output = "(5,7,9)";  % Expected output (Notice the newline character)
    if v3.x == 5 && v3.y == 7 && v3.z == 9 && strcmp(output, expected_output)
        disp('Test 1-3: Passed (Plus operator overloading)');
        pass_count = pass_count + 1;
    elseif v3.x == 5 && v3.y == 7 && v3.z == 9
        fprintf('\tExpected: (5,7,9), Got: %s\n', output);
        fprintf('\tCheck your disp function.\n');
    elseif strcmp(output, expected_output)
        fprintf('\tExpected: (5,7,9), Got: (%d,%d,%d)\n', v3.x, v3.y, v3.z);
        fprintf('\tCheck your properties.\n');
    else
        disp('Test 1-3: Failed (Plus operator overloading)');
    end
catch ME
    disp('Test 1-3: Failed (Plus operator overloading)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 1-4: Minus operator overloading (1,2,3) - (4,5,6)
try
    v4 = Vec3(1, 2, 3) - Vec3(4, 5, 6);
    output = evalc('disp(v4)');  % Capture disp output
    output = strtrim(output);
    expected_output = "(-3,-3,-3)";  % Expected output (Notice the newline character)
    if v4.x == -3 && v4.y == -3 && v4.z == -3 && strcmp(output, expected_output)
        disp('Test 1-4: Passed (Minus operator overloading)');
        pass_count = pass_count + 1;
    elseif v4.x == -3 && v4.y == -3 && v4.z == -3
        fprintf('\tExpected: (-3,-3,-3), Got: %s\n', output);
        fprintf('\tCheck your disp function.\n');
    elseif strcmp(output, expected_output)
        fprintf('\tExpected: (-3,-3,-3), Got: (%d,%d,%d)\n', v4.x, v4.y, v4.z);
        fprintf('\tCheck your properties.\n');
    else
        disp('Test 1-4: Failed (Minus operator overloading)');
    end
catch ME
    disp('Test 1-3: Failed (Plus operator overloading)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 1-5: L2 Norm
try
    norm_v3 = norm(v3);
    expected_norm = 12.4499;
    if abs(norm_v3 - expected_norm) < 1e-4
        disp('Test 1-5: Passed (norm)');
        pass_count = pass_count + 1;
    else
        disp('Test 1-5: Failed (norm)');
        fprintf('\tExpected: %d, Got: %d\n', expected_norm, norm_v3);
    end
catch ME
    disp('Test 1-5: Failed (norm)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 1-6: Inner Product
try
    ip_v3_v4 = inner_prod(v3, v4);
    expected_ip = -63;
    if abs(ip_v3_v4 - expected_ip) < 1e-4
        disp('Test 1-6: Passed (inner_prod)');
        pass_count = pass_count + 1;
    else
        disp('Test 1-6: Failed (inner_prod)');
        fprintf('\tExpected: %d, Got: %d\n', expected_ip, ip_v3_v4);
    end
catch ME
    disp('Test 1-6: Failed (inner_prod)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

if pass_count == n_subtasks
    fprintf('Congratulations! All tests in Test 1 passed!\n');
else
    fprintf('Some tests in Test 1 failed.\n');
end

% ========== Test 2 Error Handling ==========
disp("========== Test 2 Error Handling ==========");
n_subtasks = 5;
pass_count = 0;

% Test 2-1: Invalid Constructor Input (non-numeric)
try
    Vec3('a', 1, 2);
    disp('Test 2-1: Failed (non-numeric constructor input)');
catch ME % Expected error
    disp('Test 2-1: Passed (non-numeric constructor input)');
    pass_count = pass_count + 1;
end

% Test 2-2: Invalid Constructor Input (wrong number of arguments)
try
    Vec3(1, 2);  % 嘗試用錯誤數量的參數建立 Vec3
    disp('Test 2-2: Failed (wrong number of arguments)');
catch ME % Expected error
    disp('Test 2-2: Passed (wrong number of arguments)');
    pass_count = pass_count + 1;
end

% Test 2-3: Invalid Plus Operation (non-Vec3 operand)
try
    v1 = Vec3(1, 2, 3);
    result = v1 + 5;  % 嘗試與非 Vec3 對象進行加法運算
    disp('Test 2-3: Failed (non-Vec3 operand for plus)');
catch ME % Expected error
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 2-3: Passed (non-Vec3 operand for plus)");
        pass_count = pass_count + 1;
    else
        disp("Test 2-3: Failed (non-Vec3 operand for plus)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

% Test 2-4: Invalid Minus Operation (non-Vec3 operand)
try
    v1 = Vec3(1, 2, 3);
    result = v1 - 5;  % 嘗試與非 Vec3 對象進行減法運算
    disp('Test 2-4: Failed (non-Vec3 operand for minus)');
catch ME % Expected error
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 2-4: Passed (non-Vec3 operand for minus)");
        pass_count = pass_count + 1;
    else
        disp("Test 2-4: Failed (non-Vec3 operand for minus)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

% Test 2-5: Invalid Inner Product (non-Vec3 operand)
try
    v2 = Vec3(4, 5, 6);
    result = inner_prod(v2, [1, 2, 3]);
    disp('Test 2-5: Failed (non-Vec3 operand for inner_prod)');
catch ME % Expected error
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 2-5: Passed (non-Vec3 operand for inner_prod)");
        pass_count = pass_count + 1;
    else
        disp("Test 2-5: Failed (non-Vec3 operand for inner_prod)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end


if pass_count == n_subtasks
    fprintf('Congratulations! All tests in Test 2 passed!\n');
else
    fprintf('Some tests in Test 2 failed.\n');
end

fprintf('Version: 2024-11-14 02:00\n');