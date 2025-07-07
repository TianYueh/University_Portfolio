cmp ch, 91
jge tol
add ch, 32
jmp fin

tol:
sub ch, 32
jmp fin

fin:
mov ah, 1

