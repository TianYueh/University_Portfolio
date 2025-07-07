import csv
from queue import PriorityQueue
edgeFile = 'edges.csv'
heuristicFile = 'heuristic.csv'


def astar(start, end):

    # Begin your code (Part 4)
    '''
    For the A* search, I add another element for each row to 
    store their h(x), which is the straight line distance to 
    the end. In this algorithm, I open the heuristic file and
    append the node and the straight line distance to node_d
    according to the given end. Then, I assign the straight
    line distance for all rows as their h(x). After that, I 
    run the A* algorithm by each time pick up a node with 
    the shortest distance and update all the adjacent nodes
    with hp[0]-cur[7]+r[2]+r[7], where hp[0] is the 
    cummulated distance, cur[7] is parent's h(x), r[2] is the
    cost and r[7] is the new h(x).
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
            r.append(int(-1)) #h(x)
    
    node_d=[]

    with open(heuristicFile, newline='') as file2:
        fr=csv.reader(file2)
        lfr=list(fr)
        tit=lfr[0]
        lfr.pop(0) # pop out titles
        for r in lfr:
            if(end==tit[1]):
                node_d.append([int(r[0]), float(r[1])])
            elif(end==tit[2]):
                node_d.append([int(r[0]), float(r[2])])
            elif(end==tit[3]):
                node_d.append([int(r[0]), float(r[3])])

    for r in rows:
        for n in node_d:
            if(n[0]==r[1]):
                r[7]=n[1]
                break

    q=PriorityQueue()
    num_visited=0
    #First round
    for r in rows:
        if(r[0]==start and r[4]==-1):
            num_visited+=1
            r[4]=1
            #r[2]:distance, r[7]:h(x)
            q.put([(r[2]+r[7]), r])
            

    des=[]
    found=False

    while((not q.empty()) and found==False):

        hp=q.get()
        cur=hp[1]
        de=cur[1]
        
        for r in rows:
            if(r[0]==de and r[4]==-1):
                num_visited+=1
                r[4]=cur[4]+1
                r[5]=cur[0]
                r[6]=cur[1]
                q.put([(hp[0]-cur[7]+r[2]+r[7]), r])
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
    # End your code (Part 4)








if __name__ == '__main__':
    path, dist, num_visited = astar(2270143902, 1079387396)
    print(f'The number of path nodes: {len(path)}')
    print(f'Total distance of path: {dist}')
    print(f'The number of visited nodes: {num_visited}')
