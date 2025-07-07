mov ecx, 0
outer_loop:
    cmp ecx, 10
    jge final

    mov edx, 0
    jmp inner_loop

inner_loop:
    cmp edx, 10
    jge next_iter

    mov eax, [0x600000+ecx*4]
    mov ebx, [0x600000+edx*4]
    cmp eax, ebx
    jle swap
    

    inc edx
    jmp inner_loop

next_iter:
    inc ecx
    jmp outer_loop

swap:
    mov [0x600000+ecx*4], ebx
    mov [0x600000+edx*4], eax
    inc edx
    jmp inner_loop

final:
    mov eax, 1
    mov ebx, 1 


