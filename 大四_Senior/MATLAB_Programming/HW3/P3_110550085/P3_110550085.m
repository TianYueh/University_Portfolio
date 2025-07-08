% 110550085房天越

function P3_110550085()
    % open input
    fid_in = fopen('P3test.txt', 'r');
    % open output file
    fid_out = fopen('P3result.txt', 'w');
    
    % read each line
    while ~feof(fid_in)
        line = fgetl(fid_in);
        if isempty(line)
            continue; % ignore empty
        end
        
        % parse the line with given format by calling parse line function
        [num1, den1, operator, num2, den2] = parse_line(line);
        
        % call arithmetic funciton
        [result_num, result_den] = FR_arith(num1, den1, num2, den2, operator);
        
        if isempty(result_num)
            fprintf(fid_out, 'Invalid input on line: %s\n', line);
        else
            % write result to output file
            fprintf(fid_out, '%s=%d/%d\n', line, result_num, result_den);
        end
    end
    
    % close both files
    fclose(fid_in);
    fclose(fid_out);
end

function [num1, den1, operator, num2, den2] = parse_line(line)
    % extract fraction and operator with RE
    tokens = regexp(line, '(\d+)/(\d+)([\+\-\*])(\d+)/(\d+)', 'tokens');
    tokens = tokens{1}; % extract from cell array

    num1 = str2double(tokens{1});
    den1 = str2double(tokens{2});
    operator = tokens{3};
    num2 = str2double(tokens{4});
    den2 = str2double(tokens{5});
end
