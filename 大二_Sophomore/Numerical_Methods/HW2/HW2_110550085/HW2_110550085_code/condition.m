A=[0.0000000001 0;
    0 0.0000000001]

ConditionNumber(A)

function rtn=ConditionNumber(A)
Ainv=A^-1
c=norm(A)*norm(Ainv)

Ainv
end
