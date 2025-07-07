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
    x = event['x_coor']
    y = event['y_coor']
    z = event['vehicle_type']
    str1 = ("select * ,round(( 6378.138 * 2 * asin( sqrt ( pow ( sin ( ( " + str(x) + " * PI() / 180 - t1.x_coor * PI() / 180) / 2) , 2) ")
    str2 = ("                                       + COS( " + str(x) + " * PI() / 180) * COS(t1.x_coor * PI() / 180) * POW(  ")
    str3 = ("                                        SIN( ( " + str(y) + " * PI() / 180 - t1.y_coor * PI() / 180) / 2),2   ")
    str4 = ("                                        )))*1000 )) AS DISTANCE ")
    str5 = ("FROM park_info as t1 natural join park_location natural join park_name natural join park_info_realtime ")
    str6 = ("order by DISTANCE limit 5 ")
    if z == "motor":
        str7 = (" where freespace_mot > 0 ")
    elif z == "car":
        str7 = (" where freespace_car > 0 ")
    else:
        str7 = " "
    print(str1 + str2 + str3 + str4 + str5 + str6)
    cursor.execute(str1 + str2 + str3 + str4 + str5 + str7 + str6)
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
    