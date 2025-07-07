mov ebx, 0x60000f
mov edi, 2

the_loop:
    cmp ebx, 0x5fffff
    je fin

    idiv edi

    cmp edx, 0x1
    je isone

    xor edx, edx
    mov cl, 0x30
    mov [ebx], cl
    sub ebx, 0x000001
    jmp the_loop

isone:

    xor edx, edx
    mov cl, 0x31
    mov [ebx], cl
    sub ebx, 0x000001
    jmp the_loop


fin: