mov edx, [0x600000]
mov eax, edx
mov esi, [0x600004]
lea ecx, [esi]
neg ecx
imul ecx
mov edi, eax
xor edx, edx
xor eax, eax
mov ebp, [0x600008]
sub ebp, ebx
mov eax, edi
cdq
idiv ebp
mov [0x600008], eax


