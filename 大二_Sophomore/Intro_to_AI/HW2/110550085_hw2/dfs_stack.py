import csv
edgeFile = 'edges.csv'


def dfs(start, end):
    # Begin your code (Part 2)
    '''
    I choose to use stack to implement dfs, the only
    difference between the bfs and dfs is the order 
    to choose the rows to search.
    '''
    with open(edgeFile, newline='') as file:
        fr=csv.reader(file)
        rows=list(fr)
        rows.pop(0)# pop out titles
        for r in rows:
            r[0]=int(r[0]) #start node
            r[1]=int(r[1]) #end node
            r[2]=float(r[2]) #distance
            #speed limit omitted
            r.append(int(-1)) #round found
            r.append(int(-1)) #parent start
            r.append(int(-1)) #parent end
    
    q=[]

    num_visited=0
    #First round
    for r in rows:
        if(r[0]==start and r[4]==-1):
            num_visited+=1
            r[4]=1
            q.append(r)

    des=[]
    found=False

    while(found==False and len(q)!=0):
        cur=q[0]
        de=cur[1]
        q.pop(0)

        for r in rows:
            if(r[0]==de and r[4]==-1):
                num_visited+=1
                r[4]=cur[4]+1
                r[5]=cur[0]
                r[6]=cur[1]
                q.insert(0, r)
                if(r[1]==end):
                    des=r
                    found=True
                    break

    
    de=[]
    de.append(des)
    rc=des
    while(rc[0]!=start):
        for r in rows:
            if(r[0]==rc[5] and r[1]==rc[6]):
                de.append(r)
                rc=r
                break

    de.reverse()

    path=[]
    path.append(start)
    dist=0
    for r in de:
        path.append(r[1])
        dist+=r[2]

    return path, dist, num_visited
    # End your code (Part 2)


if __name__ == '__main__':
    path, dist, num_visited = dfs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
