;26 = 16 + 8 + 2
mov edx, [0x600000]
mov eax, edx
shl eax, 1
add eax, edx
shl eax, 2
add eax, edx
shl eax, 1
mov [0x600004], eax

