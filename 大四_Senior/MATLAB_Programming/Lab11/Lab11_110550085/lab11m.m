function C = lab11m(A, B)
    C = zeros(size(A), 'int32');
    for q = 1:size(B, 1)
        C(A>=B(q, 1) & A<B(q, 2)) = q;
    end
end