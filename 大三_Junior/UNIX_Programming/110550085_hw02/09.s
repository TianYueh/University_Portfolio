mov ebx, 0x5ffffe
mov edx, 0x60000f
mov edi, 0

the_loop:

    cmp ebx, 0x60000e
    je fin

    mov cl, 0
    mov cl, byte [ebx]
    


    cmp cl, 0x5c
    jg skip_conversion


    add ebx, 0x000001
    add cl, 32
    mov [edx], cl
    add edx, 0x000001
    jmp the_loop

skip_conversion:
    add ebx, 0x000001
    add edi, 1
    mov [edx], cl
    add edx, 0x000001
    jmp the_loop


fin:



