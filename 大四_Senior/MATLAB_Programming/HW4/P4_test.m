clc;
clear;

score = 0;
score_p1 = 0;
% ========== Test 1 Correctness (Scalar Operations) ==========
disp("========== Test 1 Correctness (Scalar Operations) ==========");

% Test 1-1: Constructor (1,2,3)
try
    v1 = Vec3(1, 2, 3);
    if v1.x == 1 && v1.y == 2 && v1.z == 3
        disp('Test 1-1: Passed (constructor & disp)');
        score_p1 = score_p1 + 1;
    else
        disp('Test 1-1: Failed (constructor & disp)');
        fprintf('\tExpected: (1,2,3), Got: (%d,%d,%d)\n', v1.x, v1.y, v1.z);
    end
catch ME
    disp('Test 1-1: Failed (constructor & disp)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 1-2: Constructor (0,0,0)
try
    v2 = Vec3;
    if v2.x == 0 && v2.y == 0 && v2.z == 0
        disp('Test 1-2: Passed (constructor & disp)');
        score_p1 = score_p1 + 1;
    else
        disp('Test 1-2: Failed (constructor & disp)');
        fprintf('\tExpected: (0,0,0), Got: (%d,%d,%d)\n', v2.x, v2.y, v2.z);
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
    if v3.x == 5 && v3.y == 7 && v3.z == 9
        disp('Test 1-3: Passed (Plus operator overloading)');
        score_p1 = score_p1 + 1;
    else
        disp('Test 1-3: Failed (Plus operator overloading)');
        fprintf('\tExpected: (5,7,9), Got: (%d,%d,%d)\n', v3.x, v3.y, v3.z);
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
    if v4.x == -3 && v4.y == -3 && v4.z == -3
        disp('Test 1-4: Passed (Minus operator overloading)');
        score_p1 = score_p1 + 1;
    else
        disp('Test 1-4: Failed (Minus operator overloading)');
        fprintf('\tExpected: (-3,-3,-3), Got: (%d,%d,%d)\n', v4.x, v4.y, v4.z);
    end
catch ME
    disp('Test 1-4: Failed (Minus operator overloading)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 1-5: L2 Norm
try
    norm_v3 = norm(v3);
    expected_norm = 12.4499;  % Expected value calculated manually
    if abs(norm_v3 - expected_norm) < 1e-4
        disp('Test 1-5: Passed (norm)');
        score_p1 = score_p1 + 1;
    else
        disp('Test 1-5: Failed (norm)');
        fprintf('\tExpected: %.4f, Got: %.4f\n', expected_norm, norm_v3);
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
        score_p1 = score_p1 + 1;
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

% Test 1-7: Is Zero
try
    is_zero_v1 = iszero(v1);
    is_zero_v2 = iszero(v2);
    if ~is_zero_v1 && is_zero_v2
        disp('Test 1-7: Passed (iszero)');
        score_p1 = score_p1 + 2;
    else
        disp('Test 1-7: Failed (iszero)');
    end
catch ME
    disp('Test 1-7: Failed (iszero)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 1-8: Normalize
try
    n_v1 = normalize(v1);
    expected_n_v1 = Vec3(1/sqrt(14), 2/sqrt(14), 3/sqrt(14));
    n_v2 = normalize(v2);
    if abs(n_v1.x - expected_n_v1.x) < 1e-4 && abs(n_v1.y - expected_n_v1.y) < 1e-4 && abs(n_v1.z - expected_n_v1.z) < 1e-4 && ...
       isnan(n_v2.x) && isnan(n_v2.y) && isnan(n_v2.z)
        disp('Test 1-8: Passed (normalize)');
        score_p1 = score_p1 + 2;
    else
        disp('Test 1-8: Failed (normalize)');
        fprintf('\tExpected: (%.4f,%.4f,%.4f), Got: (%.4f,%.4f,%.4f)\n', expected_n_v1.x, expected_n_v1.y, expected_n_v1.z, n_v1.x, n_v1.y, n_v1.z);
        fprintf('\tExpected: (NaN,NaN,NaN), Got: (%d,%d,%d)\n', n_v2.x, n_v2.y, n_v2.z);
    end
catch ME
    disp('Test 1-8: Failed (normalize)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 1-9: Equal operator overloading
try
    if v1 == Vec3(1, 2, 3) && ~(v1 == Vec3(1, 1, 1))
        disp('Test 1-9: Passed (equal operator overloading)');
        score_p1 = score_p1 + 2;
    else
        disp('Test 1-9: Failed (equal operator overloading)');
    end
catch ME
    disp('Test 1-9: Failed (equal operator overloading)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

fprintf('Score of Test 1: %d/12\n', score_p1);
score = score + score_p1;

% ========== Test 2 Correctness (Array Operations) ==========
score_p2 = 0;
disp("========== Test 2 Correctness (Array Operations) ==========");

% Test 2-1: Constructor ([1,2,3], [4,5,6], [7,8,9]) -> (1,4,7), (2,5,8), (3,6,9)
try
    V1 = Vec3([1,2,3], [4,5,6], [7,8,9]);
    if V1(1) == Vec3(1,4,7) && V1(2) == Vec3(2,5,8) && V1(3) == Vec3(3,6,9)
        disp('Test 2-1: Passed (constructor with array)');
        score_p2 = score_p2 + 3;
    else
        disp('Test 2-1: Failed (constructor with array)');
        fprintf('\tExpected: (1,4,7), (2,5,8), (3,6,9), Got: (%d,%d,%d), (%d,%d,%d), (%d,%d,%d)\n', V1(1).x, V1(1).y, V1(1).z, V1(2).x, V1(2).y, V1(2).z, V1(3).x, V1(3).y, V1(3).z);
    end
catch ME
    disp('Test 2-1: Failed (constructor with array)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 2-2: iszero (array)
try
    V2 = [Vec3(0, 0, 0) V1];
    V2 = reshape(V2, 2, 2);
    V2_iszero = iszero(V2);
    %  1     0
    %  0     0
    if isequal(size(V2_iszero), size(V2)) && ...
        V2_iszero(1,1) && ~V2_iszero(1,2) && ~V2_iszero(2,1) && ~V2_iszero(2,2)
        disp('Test 2-2: Passed (iszero with array)');
        score_p2 = score_p2 + 3;
    else
        disp('Test 2-2: Failed (iszero with array)');
        if ~isequal(size(V2_iszero), size(V2))
            fprintf('\tExpected Size: (%d,%d), Got: (%d,%d)\n', size(V2,1), size(V2,2), size(V2_iszero,1), size(V2_iszero,2));
        end
    end
catch ME
    disp('Test 2-2: Failed (iszero with array)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 2-3: Plus operator overloading (array)
try
    V3 = [V1 Vec3(1, 2, 3)];
    V3 = reshape(V3, 2, 2);
    V5 = V2 + V3;
    % (1, 4, 7)
    % (3, 9, 15)
    % (5, 11, 17)
    % (2, 4, 6)
    if isequal(size(V5), size(V2)) && ...
        V5(1) == Vec3(1,4,7) && V5(2) == Vec3(3,9,15) && V5(3) == Vec3(5,11,17) && V5(4) == Vec3(4,8,12)
        disp('Test 2-3: Passed (Plus operator overloading with array)');
        score_p2 = score_p2 + 3;
    else
        disp('Test 2-3: Failed (Plus operator overloading with array)');
        if ~isequal(size(V5), size(V2))
            fprintf('\tExpected Size: (%d,%d), Got: (%d,%d)\n', size(V2,1), size(V2,2), size(V5,1), size(V5,2));
        end
        if ~(V5(1) == Vec3(1,4,7))
            fprintf('\tExpected: (1,4,7), Got: (%d,%d,%d)\n', V5(1).x, V5(1).y, V5(1).z);
        end
        if ~(V5(2) == Vec3(3,9,15))
            fprintf('\tExpected: (3,9,15), Got: (%d,%d,%d)\n', V5(2).x, V5(2).y, V5(2).z);
        end
        if ~(V5(3) == Vec3(5,11,17))
            fprintf('\tExpected: (5,11,17), Got: (%d,%d,%d)\n', V5(3).x, V5(3).y, V5(3).z);
        end
        if ~(V5(4) == Vec3(4,8,12))
            fprintf('\tExpected: (4,8,12), Got: (%d,%d,%d)\n', V5(4).x, V5(4).y, V5(4).z);
        end
    end
catch ME
    disp('Test 2-3: Failed (Plus operator overloading with array)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 2-4: Minus operator overloading (array)
try
    V4 = [V1 Vec3(-1, -2, -3)];
    V4 = reshape(V4, 2, 2);
    V6 = V2 - V4;
    % (-1, -4, -7)
    % (-1, -1, -1)
    % (-1, -1, -1)
    % (4, 8, 12)
    if isequal(size(V6), size(V2)) && ...
        V6(1) == Vec3(-1,-4,-7) && V6(2) == Vec3(-1,-1,-1) && V6(3) == Vec3(-1,-1,-1) && V6(4) == Vec3(4,8,12)
        disp('Test 2-4: Passed (Minus operator overloading with array)');
        score_p2 = score_p2 + 3;
    else
        disp('Test 2-4: Failed (Minus operator overloading with array)');
        if ~isequal(size(V6), size(V2))
            fprintf('\tExpected Size: (%d,%d), Got: (%d,%d)\n', size(V2,1), size(V2,2), size(V6,1), size(V6,2));
        end
        if ~(V6(1) == Vec3(-1,-4,-7))
            fprintf('\tExpected: (-1,-4,-7), Got: (%d,%d,%d)\n', V6(1).x, V6(1).y, V6(1).z);
        end
        if ~(V6(2) == Vec3(-1,-1,-1))
            fprintf('\tExpected: (-1,-1,-1), Got: (%d,%d,%d)\n', V6(2).x, V6(2).y, V6(2).z);
        end
        if ~(V6(3) == Vec3(-1,-1,-1))
            fprintf('\tExpected: (-1,-1,-1), Got: (%d,%d,%d)\n', V6(3).x, V6(3).y, V6(3).z);
        end
        if ~(V6(4) == Vec3(4,8,12))
            fprintf('\tExpected: (4,8,12), Got: (%d,%d,%d)\n', V6(4).x, V6(4).y, V6(4).z);
        end
    end
catch ME
    disp('Test 2-3: Failed (Plus operator overloading with array)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 2-5: Equal operator overloading (array)
try
    is_equal_V5_V6 = V5 == V6;
    % 0     0
    % 0     1
    if isequal(size(is_equal_V5_V6), size(V5)) && ...
        ~is_equal_V5_V6(1) && ~is_equal_V5_V6(2) && ~is_equal_V5_V6(3) && is_equal_V5_V6(4)
        disp('Test 2-5: Passed (Equal operator overloading with array)');
        score_p2 = score_p2 + 3;
    else
        disp('Test 2-5: Failed (Equal operator overloading with array)');
    end
catch ME
    disp('Test 2-5: Failed (Equal operator overloading with array)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 2-6: Inner Product (array)
try
    ip_V5_V6 = inner_prod(V5, V6);
    % -66   -33
    % -27   224
    if isequal(size(ip_V5_V6), size(V5)) && ...
        ip_V5_V6(1) == -66 && ip_V5_V6(2) == -27 && ip_V5_V6(3) == -33 && ip_V5_V6(4) == 224
        disp('Test 2-6: Passed (Inner Product with array)');
        score_p2 = score_p2 + 3;
    else
        disp('Test 2-6: Failed (Inner Product with array)');
    end
catch ME
    disp('Test 2-6: Failed (Inner Product with array)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 2-7: L2 Norm (array)
try
    l2_norm_V5 = norm(V5);
%     8.1240   20.8567
%    17.7482   14.9666
    if isequal(size(l2_norm_V5), size(V5)) && ...
        abs(l2_norm_V5(1) - 8.1240) < 1e-4 && abs(l2_norm_V5(2) - 17.7482) < 1e-4 && abs(l2_norm_V5(3) - 20.8567) < 1e-4 && abs(l2_norm_V5(4) - 14.9666) < 1e-4
        disp('Test 2-7: Passed (L2 Norm with array)');
        score_p2 = score_p2 + 3;
    else
        disp('Test 2-7: Failed (L2 Norm with array)');
    end
catch ME
    disp('Test 2-7: Failed (L2 Norm with array)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end


% Test 2-8: Normalize (array)
try
    norm_V5 = normalize(V5);
    % (0.123091, 0.492366, 0.86164)
    % (0.169031, 0.507093, 0.845154)
    % (0.239732, 0.52741, 0.815088)
    % (0.267261, 0.534522, 0.801784)
    if isequal(size(norm_V5), size(V5)) && ...
        abs(norm_V5(1).x - 0.123091) < 1e-4 && abs(norm_V5(1).y - 0.492366) < 1e-4 && abs(norm_V5(1).z - 0.86164) < 1e-4 && ...
        abs(norm_V5(2).x - 0.169031) < 1e-4 && abs(norm_V5(2).y - 0.507093) < 1e-4 && abs(norm_V5(2).z - 0.845154) < 1e-4 && ...
        abs(norm_V5(3).x - 0.239732) < 1e-4 && abs(norm_V5(3).y - 0.52741) < 1e-4 && abs(norm_V5(3).z - 0.815088) < 1e-4 && ...
        abs(norm_V5(4).x - 0.267261) < 1e-4 && abs(norm_V5(4).y - 0.534522) < 1e-4 && abs(norm_V5(4).z - 0.801784) < 1e-4
        disp('Test 2-8: Passed (Normalize with array)');
        score_p2 = score_p2 + 3;
    else
        disp('Test 2-8: Failed (Normalize with array)');
    end
catch ME
    disp('Test 2-8: Failed (Normalize with array)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

fprintf('Score of Test 2: %d/24\n', score_p2);
score = score + score_p2;


% ========== Test 3 Error Handling (basic)==========
score_p3 = 0;
disp("========== Test 3 Error Handling (basic) ==========");

% Test 3-1: Invalid Constructor Input (non-numeric)
try
    Vec3('a', 1, 2);
    disp('Test 3-1: Failed (non-numeric constructor input)');
catch ME
    disp('Test 3-1: Passed (non-numeric constructor input)');
    score_p3 = score_p3 + 2;
end

% Test 3-2: Invalid Constructor Input (wrong number of arguments)
try
    Vec3(1, 2);
    disp('Test 3-2: Failed (wrong number of arguments)');
catch ME
    disp('Test 3-2: Passed (wrong number of arguments)');
    score_p3 = score_p3 + 2;
end

% Test 2-3: Invalid Plus Operation (non-Vec3 operand)
try
    v1 = Vec3(1, 2, 3);
    result = v1 + 5;
    disp('Test 3-3: Failed (non-Vec3 operand for plus)');
catch ME
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 3-3: Passed (non-Vec3 operand for plus)");
        score_p3 = score_p3 + 2;
    else
        disp("Test 3-3: Failed (non-Vec3 operand for plus)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

% Test 3-4: Invalid Minus Operation (non-Vec3 operand)
try
    v1 = Vec3(1, 2, 3);
    result = v1 - 5;
    disp('Test 3-4: Failed (non-Vec3 operand for minus)');
catch ME
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 3-4: Passed (non-Vec3 operand for minus)");
        score_p3 = score_p3 + 2;
    else
        disp("Test 3-4: Failed (non-Vec3 operand for minus)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

% Test 3-5: Invalid Inner Product (non-Vec3 operand)
try
    v2 = Vec3(4, 5, 6);
    result = inner_prod(v2, [1, 2, 3]);
    disp('Test 3-5: Failed (non-Vec3 operand for inner_prod)');
catch ME
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 3-5: Passed (non-Vec3 operand for inner_prod)");
        score_p3 = score_p3 + 2;
    else
        disp("Test 3-5: Failed (non-Vec3 operand for inner_prod)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

fprintf('Score of Test 3: %d/10\n', score_p3);
score = score + score_p3;

% ========== Test 4 Error Handling (array)==========
score_p4 = 0;
disp("========== Test 4 Error Handling (array) ==========");

% 1. Invalid Is Zero Operation (non-Vec3 operand)
% 2. Invalid Equal Operation (non-Vec3 operand)
% 3. Invalid Constructor Input (different size)
% 4. Invalid Operand for Plus (different size)
% 5. Invalid Operand for Minus (different size)
% 6. Invalid Operand for Inner Product (different size)

% Test 4-1: Invalid Is Zero Operation (non-Vec3 operand)
try
    result = is_zero([1, 2, 3]);
    disp('Test 4-1: Failed (non-Vec3 operand for is_zero)');
catch ME
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 4-1: Passed (non-Vec3 operand for is_zero)");
        score_p4 = score_p4 + 3;
    else
        disp("Test 4-1: Failed (non-Vec3 operand for is_zero)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

% Test 4-2: Invalid Equal Operation (non-Vec3 operand)
try
    result = [1, 2, 3] == Vec3(1, 2, 3);
    disp('Test 4-2: Failed (non-Vec3 operand for equal)');
catch ME
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 4-2: Passed (non-Vec3 operand for equal)");
        score_p4 = score_p4 + 3;
    else
        disp("Test 4-2: Failed (non-Vec3 operand for equal)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

% Test 4-3: Invalid Constructor Input (different size)
try
    T3 = Vec3([1, 2, 3], [4, 5, 6], [7, 8, 9, 10]);
    disp('Test 4-3: Failed (different size for constructor)');
catch ME
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 4-3: Passed (different size for constructor)");
        score_p4 = score_p4 + 3;
    else
        disp("Test 4-3: Failed (different size for constructor)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

% Test 4-4: Invalid Operand for Plus (different size)
try
    T1 = Vec3([1, 2, 3], [4, 5, 6], [7, 8, 9]);
    T2 = Vec3([1, 2, 3], [4, 5, 6]);
    result = T1 + T2;
    disp('Test 4-4: Failed (different size for plus)');
catch ME
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 4-4: Passed (different size for plus)");
        score_p4 = score_p4 + 3;
    else
        disp("Test 4-4: Failed (different size for plus)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

% Test 4-5: Invalid Operand for Minus (different size)
try
    T1 = Vec3([1, 2, 3], [4, 5, 6], [7, 8, 9]);
    T2 = Vec3([1, 2, 3], [4, 5, 6]);
    result = T1 - T2;
    disp('Test 4-5: Failed (different size for minus)');
catch ME
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 4-5: Passed (different size for minus)");
        score_p4 = score_p4 + 3;
    else
        disp("Test 4-5: Failed (different size for minus)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

% Test 4-6: Invalid Operand for Inner Product (different size)
try
    T1 = Vec3([1, 2, 3], [4, 5, 6], [7, 8, 9]);
    T2 = Vec3([1, 2, 3], [4, 5, 6]);
    result = inner_prod(T1, T2);
    disp('Test 4-6: Failed (different size for inner_prod)');
catch ME
    % 1. validateattributes / 2. 手動檢查 / 3. 其他但非原本的報錯
    if contains(ME.identifier, 'expectedInteger') || strcmp(ME.identifier, '') || ~contains(ME.identifier, 'structRefFromNonStruct')
        disp("Test 4-6: Passed (different size for inner_prod)");
        score_p4 = score_p4 + 3;
    else
        disp("Test 4-6: Failed (different size for inner_prod)");
        if ~isempty(ME.stack)
            fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
        end
        fprintf('\tIdentifier: %s\n', ME.identifier);
        fprintf('\tMessage: %s\n', ME.message);
    end
end

fprintf('Score of Test 4: %d/18\n', score_p4);
score = score + score_p4;

% ========== Test 5: Boardcast (Bonus)=========
bonus = 0;
disp("========== Test 5: Boardcast (Bonus) ==========");

% Test 5-1: Broadcast Plus
try
    v1 = Vec3(1, 2, 3);
    v2 = Vec3([1, 2, 3], [4, 5, 6], [7, 8, 9]);
    v3 = v1 + v2;
    % (2, 6, 10)
    % (3, 7, 11)
    % (4, 8, 12)
    if isequal(size(v3), size(v2)) && ...
        v3(1) == Vec3(2, 6, 10) && v3(2) == Vec3(3, 7, 11) && v3(3) == Vec3(4, 8, 12)
        disp('Test 5-1: Passed (broadcast plus)');
        bonus = bonus + 4;
    else
        disp('Test 5-1: Failed (broadcast plus)');
    end
catch ME
    disp('Test 5-1: Failed (broadcast plus)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 5-2: Broadcast Minus
try
    v1 = Vec3(1, 2, 3);
    v2 = Vec3([1, 2, 3], [4, 5, 6], [7, 8, 9]);
    v4 = v1 - v2;
    % (0, -2, -4)
    % (-1, -3, -5)
    % (-2, -4, -6)
    if isequal(size(v4), size(v2)) && ...
        v4(1) == Vec3(0, -2, -4) && v4(2) == Vec3(-1, -3, -5) && v4(3) == Vec3(-2, -4, -6)
        disp('Test 5-2: Passed (broadcast minus)');
        bonus = bonus + 4;
    else
        disp('Test 5-2: Failed (broadcast minus)');
    end
catch ME
    disp('Test 5-2: Failed (broadcast minus)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

% Test 5-3: Broadcast Inner Product
try
    v1 = Vec3(1, 2, 3);
    v2 = Vec3([1, 2, 3], [4, 5, 6], [7, 8, 9]);
    v5 = inner_prod(v1, v2);
    % 30
    % 36
    % 42
    if isequal(size(v5), size(v2)) && ...
        v5(1) == 30 && v5(2) == 36 && v5(3) == 42
        disp('Test 5-3: Passed (broadcast inner_prod)');
        bonus = bonus + 4;
    else
        disp('Test 5-3: Failed (broadcast inner_prod)');
    end
catch ME
    disp('Test 5-3: Failed (broadcast inner_prod)');
    if ~isempty(ME.stack)
        fprintf('\tError occurred in %s at line %d\n', ME.stack(1).name, ME.stack(1).line);
    end
    fprintf('\tIdentifier: %s\n', ME.identifier);
    fprintf('\tMessage: %s\n', ME.message);
end

fprintf('Score of Bonus: %d/12\n', bonus);
score = score + bonus;

disp('--------------------------------');
fprintf('Total Score: %d/36 + %d/28 + %d/12 = %d/76\n', score_p1 + score_p2, score_p3 + score_p4, bonus, score);

fprintf('Last Updated: 2024-12-05 13:20\n');