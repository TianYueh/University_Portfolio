% 110550085房天越

function [numerator, denominator] = FR_arith(num1, den1, num2, den2, operator)
    % check if divide by 0
    if den1 == 0 || den2 == 0
        numerator = [];
        denominator = [];
        return;
    end
    
    % do calculation by operator

    if operator == '+'
        [numerator, denominator] = FR_add(num1, den1, num2, den2);
    elseif operator == '-'
        [numerator, denominator] = FR_subtract(num1, den1, num2, den2);
    elseif operator == '*'
        [numerator, denominator] = FR_multiply(num1, den1, num2, den2);
    else 
        numerator = [];
        denominator = [];
    end

    
    % reduce the fraction
    [numerator, denominator] = reduce_fraction(numerator, denominator);
end

function [num, den] = FR_add(num1, den1, num2, den2)
    % add
    num = num1 * den2 + num2 * den1;
    den = den1 * den2;
end

function [num, den] = FR_subtract(num1, den1, num2, den2)
    % subtract
    num = num1 * den2 - num2 * den1;
    den = den1 * den2;
end

function [num, den] = FR_multiply(num1, den1, num2, den2)
    % multiply
    num = num1 * num2;
    den = den1 * den2;
end

function [num, den] = reduce_fraction(num, den)
    % reduce to simplest form
    gcd_val = gcd(num, den);
    num = num / gcd_val;
    den = den / gcd_val;
    
    % make sure dominator positive
    if den < 0
        num = -num;
        den = -den;
    end
end
