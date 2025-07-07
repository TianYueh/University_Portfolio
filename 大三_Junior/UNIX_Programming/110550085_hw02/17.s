mov edi, 1
mov esi, -1

test eax, eax
jns aestp
mov [0x600000], esi
jmp testb


aestp:
mov [0x600000], edi
jmp testb

testb:
test ebx, ebx
jns bestp
mov [0x600004], esi
jmp testc

bestp:
mov [0x600004], edi
jmp testc

testc:
test ecx, ecx
jns cestp
mov [0x600008], esi
jmp testd

cestp:
mov [0x600008], edi
jmp testd

testd:
test edx, edx
jns destp
mov [0x60000c], esi
jmp fini

destp:
mov [0x60000c], edi
jmp fini

fini:
mov eax, 1


