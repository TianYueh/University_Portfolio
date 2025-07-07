import urllib.request as request
import json
import psycopg2 
import sys
from datetime import datetime


def lambda_handler(event, context):
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
    x = event['parkinglot']
    str1 = "with update_hours as( "
    str2 = "select *, extract(hour from update_time) as hours "
    str3 = "from park_info_realtime_old), "
    str4 = "park_avg as( "
    str5 = "select park_id ,round(avg(freespace_car)) as avgfreecar, round(avg(freespace_mot)) as avgfreemotor ,hours "
    str6 = "from update_hours " 
    str7 = "group by hours ,park_id ) "
    str8 = "select avgfreecar, avgfreemotor , totalspace_car, totalspace_mot , hours " 
    str9 = "from park_avg natural join park_info natural join park_name "
    str10 = "where park_name = '"
    str11 = "' order by hours "
    cursor.execute(str1 + str2 + str3 + str4 + str5 + str6 + str7 + str8 + str9 + str10 + x + str11)
    rows = cursor.fetchall()
    temp = [[0 for col in range(len(rows[0]))] for row in range(len(rows))]
    for i in range(len(rows)):
        for j in range(len(rows[0])):
            if type(rows[i][j]) == datetime:
                temp[i][j] = rows[i][j].strftime("%Y-%m-%d %H:%M:%S")
            else:
                temp[i][j] = rows[i][j]
    cursor.close()
    conn.close()
    return {
        "statuscode": 200,
        "query": temp
    }
    