mov edi, [0x600000]
mov ebx, [0x600004]
mov ecx, 5
mov eax, edi
mul ecx
sub ebx, 3
xor edx, edx
div ebx
mov [0x600008], eax




