mov     edx, [0x600000]
mov     eax, edx
imul    eax, 5
lea     ecx, [eax]
neg     ecx
mov     eax, [0x600004]
neg     eax
mov     esi, [0x600008]
cdq
idiv    esi
mov     edi, edx
mov     eax, ecx
cdq
idiv    edi
mov     [0x60000c], eax
xor     eax, eax









