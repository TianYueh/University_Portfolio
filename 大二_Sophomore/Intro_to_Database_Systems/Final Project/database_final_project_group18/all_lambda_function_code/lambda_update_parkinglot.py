import urllib.request as request
import json
import psycopg2 
import sys


def lambda_handler(event, context): 
    src = "https://hispark.hccg.gov.tw/OpenData/GetParkInfo"
    request.urlopen(src)
    with request.urlopen(src) as response:
        data = json.load(response)
    
    
    host = "dbfinalproject.ccmebuq8re9n.us-east-1.rds.amazonaws.com"
    dbname = "dbfinalproject"
    user = "postgres"
    password = "nycu_postgres"
    sslmode = "allow"
    conn_string = "host={0} user={1} dbname={2} password={3} sslmode={4}".format(host, user, dbname, password, sslmode)
    conn = psycopg2.connect(conn_string)
    print("Connection established")
    conn.autocommit = True
    
    cursor = conn.cursor()
    cursor.execute("select * from park_info_realtime_temp")
    rows = cursor.fetchall()
    for row in rows:
        try:
            cursor.execute("insert into park_info_realtime_old(park_id, freespace_car, freespace_mot, update_time) values (%s, %s, %s, %s)", (row[0],row[1],row[2],row[3]))
        except Exception as e:
            print(e)
            conn.rollback()
    cursor.execute("truncate table park_info_realtime_temp")
    cursor.execute("insert into park_info_realtime_temp select * from park_info_realtime")
    cursor.execute("truncate table park_info_realtime")
    for parking in data:
        cursor.execute("insert into park_info_realtime(park_id,freespace_car,freespace_mot,update_time) values (%s,%s,%s,%s)",(int(parking["PARKNO"]),int(parking["FREESPACE"]),int(parking["FREESPACEMOT"]),str(parking["UPDATETIME"])))
    
    cursor.close()
    conn.close()
    return {
        'statuscode': 1
    }