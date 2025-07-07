mov rdi, 18
call recur
mov rsi, 2
inc rax
cdq
idiv rsi
jmp exit

recur:
    push rcx 
    
    cmp rdi, 0
    jle base_case_0
    cmp rdi, 1
    je base_case_1

    push rdi
    dec rdi
    call recur
    mov r11, rax

    pop rdi
    dec rdi
    call recur
    imul rax, 3
    add rax, r11

    pop rcx ; Restore rcx
    ret

base_case_0:
    xor rax, rax
    pop rcx ; Restore rcx
    ret

base_case_1:
    mov rax, 1
    pop rcx ; Restore rcx
    ret

exit: