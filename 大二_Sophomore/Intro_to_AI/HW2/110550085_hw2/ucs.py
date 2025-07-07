import csv
from queue import PriorityQueue
edgeFile = 'edges.csv'


def ucs(start, end):
    # Begin your code (Part 3)
    '''
    UCS algorithm is kind of similar to Dijkstra, in my code,
    I use the PriorityQueue q to determine the node to search
    with the value of distance, every time, I pick up the one
    with the smallest distance and for every adjacent nodes, 
    I put a new one with the distance of the sum of the 
    current node and the distance to that node.
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
    
    q=PriorityQueue()

    num_visited=0
    #First round
    for r in rows:
        if(r[0]==start and r[4]==-1):
            num_visited+=1
            r[4]=1
            q.put([r[2], r])

    des=[]
    found=False

    while(found==False and not(q.empty())):

        hp=q.get()
        cur=hp[1]
        de=cur[1]
        
        for r in rows:
            if(r[0]==de and r[4]==-1):
                num_visited+=1
                r[4]=cur[4]+1
                r[5]=cur[0]
                r[6]=cur[1]
                q.put([hp[0]+r[2], r])
                if(r[1]==end):
                    des=r
                    found=True
                    break


    de=[]
    de.append(des)
    rc=des
    # find the route back then reverse
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
    # End your code (Part 3)


if __name__ == '__main__':
    path, dist, num_visited = ucs(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
