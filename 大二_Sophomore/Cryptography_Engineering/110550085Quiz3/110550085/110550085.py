


def iccount(file, length):
    group={}
    cnt=[0]*length
    for i in range(length):
        group[i]=[0]*26

    for i in range(len(file)):
        if(ord(file[i])-ord('A')<=26 and ord(file[i])-ord('A')>=0):
            #print(i%length)
            #print(ord(file[i])-ord('A'))
            group[i%length][ord(file[i])-ord('A')] +=1
            cnt[i%length]+=1

    ic=0
    for i in range (length):
        for l in range(26):
            ic+=(group[i][l]*group[i][l]-1)/cnt[i]/(cnt[i]-1)
    ic/=length
    return ic

def find_length(t):
    f=open(t,'r')
    msg=f.read()
    #print('YOYO')
    #print(os.getcwd())
    md=999
    for i in range (4,8):
        d=abs(iccount(msg, i)-0.066)
        #print(d)
        if(d<md):
            md=d
            length=i
    print(t,":", length)
    return length

def find_key(t, num):
    f=open(t, 'r')
    msg=f.read()
    group={}
    for i in range(num):
        group[i]=[0]*26

    for i in range(len(msg)):
        if(ord(msg[i])-ord('A')>=0 and ord(msg[i])-ord('A')<26):
            group[i%num][ord(msg[i])-ord('A')] += 1
    
    str=""
    for i in range(num):
        str += chr(fk(group,i)+ord('A'))

    print(t,":",str)
    return str

freq=[8.167,1.492,2.782,4.253,12.702,
      2.228,2.015,6.094,6.966,0.153,
      0.772,4.025,2.406,6.749,7.507,
      1.929,0.095,5.987,6.327,9.056,
      2.758,0.978,2.360,0.150,1.974,0.074]

def fk(group, index):
    maxi=0
    mk=0
    for k in range(26):
        val=0
        for l in range(26):
            val+=group[index][(l+k)%26]*freq[l]
        if(val>maxi):
            maxi=val
            mk=k
    return mk

def turn_plain(t, num, key):
    f=open(t, 'r')
    msg=f.read()
    str=""
    for i in range(len(msg)):
        #x=ord(msg[i])-ord('A')+ord(key[i%num])
        str+=chr( ( ord(msg[i]) -ord(key[i%num]) )%26+ord('A'))
    print(t, ":", str)
    return str


f=open('110550085_msg1.txt', 'w')
ans11=find_length('msg1.txt')
f.write(str(ans11))
f.write('\n')
ans21=find_key('msg1.txt', ans11)
f.write(str(ans21))
f.write('\n')
ans31=turn_plain('msg1.txt', ans11, ans21)
f.write(str(ans31))
f.write('\n')
f.close()

f=open('110550085_msg2.txt', 'w')
ans12=find_length('msg2.txt')
f.write(str(ans12))
f.write('\n')
ans22=find_key('msg2.txt', ans12)
f.write(str(ans22))
f.write('\n')
ans32=turn_plain('msg2.txt', ans12, ans22)
f.write(str(ans32))
f.write('\n')
f.close()









