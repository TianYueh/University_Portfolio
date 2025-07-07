mov si, [0x600000]
mov bp, [0x600001]
shr ax, 5
and ax, 0x007f
mov [0x600000], ax
mov [0x600001], bp
