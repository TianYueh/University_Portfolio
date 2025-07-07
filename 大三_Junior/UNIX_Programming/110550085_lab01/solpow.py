#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import base64
import hashlib
import time
import sys
import re
from pwn import *

number_models = {
    '0': [' ┌───┐ ', ' │   │ ', ' │   │ ', ' │   │ ', ' └───┘ '],
    '1': ['  ─┐   ', '   │   ', '   │   ', '   │   ', '  ─┴─  '],
    '2': [' ┌───┐ ', '     │ ', ' ┌───┘ ', ' │     ', ' └───┘ '],
    '3': [' ┌───┐ ', '     │ ', '  ───┤ ', '     │ ', ' └───┘ '],
    '4': [' │   │ ', ' │   │ ', ' └───┤ ', '     │ ', '     │ '],
    '5': [' ┌──── ', ' │     ', ' └───┐ ', '     │ ', ' └───┘ '],
    '6': [' ┌───┐ ', ' │     ', ' ├───┐ ', ' │   │ ', ' └───┘ '],
    '7': [' ┌───┐ ', ' │   │ ', '     │ ', '     │ ', '     │ '],
    '8': [' ┌───┐ ', ' │   │ ', ' ├───┤ ', ' │   │ ', ' └───┘ '],
    '9': [' ┌───┐ ', ' │   │ ', ' └───┤ ', '     │ ', ' └───┘ '],
    '+': ['       ', '   │   ', ' ──┼── ', '   │   ', '       '],
    '/': ['       ', '   •   ', ' ───── ', '   •   ', '       '],
    '*': ['       ', '  ╲ ╱  ', '   ╳   ', '  ╱ ╲  ', '       ']
}

def solve_pow(r):
    prefix = r.recvline().decode().split("'")[1];
    print(time.time(), "solving pow ...");
    solved = b''
    for i in range(1000000000):
        h = hashlib.sha1((prefix + str(i)).encode()).hexdigest();
        if h[:6] == '000000':
            solved = str(i).encode();
            print("solved =", solved);
            break;
    print(time.time(), "done.");
    r.sendlineafter(b'string S: ', base64.b64encode(solved));
    
def match_number_with_model(received_number, number_models):
    matched_number = ''
    for digit, model in number_models.items():
        if all(received_line == model_line for received_line, model_line in zip(received_number, model)):
            matched_number = digit
            break
    return matched_number

def match_lists_with_models(split_lists, number_models):
    matched_numbers = []
    for lst in split_lists:
        matched_number = match_number_with_model(lst, number_models)
        matched_numbers.append(matched_number)
    return matched_numbers

def evaluate_expression(expression):
    try:
        # Ensure that the expression contains only allowed characters
        allowed_chars = set('0123456789+*/ ')
        if not set(expression).issubset(allowed_chars):
            raise ValueError("Invalid characters in the expression")

        # Evaluate the expression
        result = eval(expression)

        # Ensure that the result is a single number
        if not isinstance(result, (int, float)):
            raise ValueError("Invalid expression result")

        return int(result)
    except Exception as e:
        print("Error:", e)
        return None
    
def extract_challenge_count(message):
    # Define the pattern to match the number between "the" and "challenges"
    pattern = r'the (\d+) challenges'
    
    # Search for the pattern in the message
    match = re.search(pattern, message)
    
    if match:
        # Extract and return the matched number
        return int(match.group(1))
    else:
        # Return None if no match is found
        return None

if __name__ == "__main__":
    r = None
    if len(sys.argv) == 2:
        r = remote('localhost', int(sys.argv[1]))
    elif len(sys.argv) == 3:
        r = remote(sys.argv[2], int(sys.argv[1]))
    else:
        r = process('./pow.py')
    
    solve_pow(r);
    
    #-----
    received_message = r.recvuntil(b'?').strip().decode()
    print("Received message:", received_message)
    
    encoded_part = received_message.split(' =')[-2].strip()
    true_encoded_part = encoded_part.split(': ')[-1].strip()
    
    print(true_encoded_part)
    
    decoded_part = base64.b64decode(true_encoded_part).decode()
    decoded_list = decoded_part.split('\n')
    
    print(decoded_part)
    print(decoded_list)
    
    split_lists = [[] for _ in range(7)]

    # Iterate over each line in the decoded list
    for line in decoded_list:
        # Split each line into segments corresponding to each number
        segments = [line[i:i+7] for i in range(0, len(line), 7)]
    
        # Append each segment to the corresponding split list
        for i, segment in enumerate(segments):
            split_lists[i].append(segment)

    # Print the split lists for each number
    for i, lst in enumerate(split_lists):
        
        print(f"List for number {i+1}: {lst}")
        
    print(split_lists)
    matched_numbers = match_lists_with_models(split_lists, number_models)

    # Print the matched numbers
    for i, matched_number in enumerate(matched_numbers):
        print(f"Matched number for list {i+1}: {matched_number}")
        
    print(matched_numbers)
    
    # Convert the list of matched numbers to a single string
    matched_numbers_string = ''.join(matched_numbers)

    # Print the string representation of matched numbers
    print("Matched numbers:", matched_numbers_string)
    
    result = evaluate_expression(matched_numbers_string)
    if result is not None:
        print("Result:", result)
            
    r.sendline(str(result))
    
    # Receive the message
    
    #received_message = "Please complete the 165 challenges in a limited time."
    challenge_count = extract_challenge_count(received_message)
    if challenge_count is not None:
        print("Number of challenges:", challenge_count)
    else:
        print("No match found.")
    
    
    
    for i in range(challenge_count-1):
        received_message = r.recvuntil(b'?').strip().decode()
        print("Received message:", received_message)
    
        encoded_part = received_message.split(' =')[-2].strip()
        true_encoded_part = encoded_part.split(': ')[-1].strip()
    
        print(true_encoded_part)
    
        decoded_part = base64.b64decode(true_encoded_part).decode()
        decoded_list = decoded_part.split('\n')
    
        print(decoded_part)
        #print(decoded_array)
        print(decoded_list)
    
        split_lists = [[] for _ in range(7)]

        # Iterate over each line in the decoded list
        for line in decoded_list:
            # Split each line into segments corresponding to each number
            segments = [line[i:i+7] for i in range(0, len(line), 7)]
    
            # Append each segment to the corresponding split list
            for i, segment in enumerate(segments):
                split_lists[i].append(segment)

        # Print the split lists for each number
        for i, lst in enumerate(split_lists):
        
            print(f"List for number {i+1}: {lst}")
        
        print(split_lists)
        matched_numbers = match_lists_with_models(split_lists, number_models)

        # Print the matched numbers
        for i, matched_number in enumerate(matched_numbers):
            print(f"Matched number for list {i+1}: {matched_number}")
        
        print(matched_numbers)
    
        # Convert the list of matched numbers to a single string
        matched_numbers_string = ''.join(matched_numbers)

        # Print the string representation of matched numbers
        print("Matched numbers:", matched_numbers_string)
    
        result = evaluate_expression(matched_numbers_string)
        if result is not None:
            print("Result:", result)
            
        r.sendline(str(result))

    
    
    
        with open("received_message.txt", "w") as f:
            f.write(received_message)
    
    
    
        
    r.interactive();
    
    '''
    received_message = r.recvuntil(b'?').strip().decode()
    print("Received message:", received_message)
    
    encoded_part = received_message.split(' =')[-2].strip()
    true_encoded_part = encoded_part.split(': ')[-1].strip()
    
    print(true_encoded_part)
    
    decoded_part = base64.b64decode(true_encoded_part).decode()
    decoded_list = decoded_part.split('\n')
    
    print(decoded_part)
    #print(decoded_array)
    print(decoded_list)
    
    split_lists = [[] for _ in range(7)]

    # Iterate over each line in the decoded list
    for line in decoded_list:
        # Split each line into segments corresponding to each number
        segments = [line[i:i+7] for i in range(0, len(line), 7)]
    
        # Append each segment to the corresponding split list
        for i, segment in enumerate(segments):
            split_lists[i].append(segment)

    # Print the split lists for each number
    for i, lst in enumerate(split_lists):
        
        print(f"List for number {i+1}: {lst}")
        
    print(split_lists)
    matched_numbers = match_lists_with_models(split_lists, number_models)

    # Print the matched numbers
    for i, matched_number in enumerate(matched_numbers):
        print(f"Matched number for list {i+1}: {matched_number}")
        
    print(matched_numbers)
    
    # Convert the list of matched numbers to a single string
    matched_numbers_string = ''.join(matched_numbers)

    # Print the string representation of matched numbers
    print("Matched numbers:", matched_numbers_string)
    
    result = evaluate_expression(matched_numbers_string)
    if result is not None:
        print("Result:", result)

    
    
    
    with open("received_message.txt", "w") as f:
        f.write(received_message)
    
    
    
        
    #r.interactive();
    '''
    
    r.close();

# vim: set tabstop=4 expandtab shiftwidth=4 softtabstop=4 number cindent fileencoding=utf-8 :
