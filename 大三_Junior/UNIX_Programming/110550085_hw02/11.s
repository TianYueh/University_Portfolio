mov eax, [0x600000]
mov ebx, [0x600004]
mov ecx, [0x600008]
neg eax
imul ebx
add eax, ecx