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
    x = event['location']
    z = event["vehicle_type"]
    current_time = datetime.now()
    stri1 = "insert into history_search(location_name,search_time) values('"
    stri2 = "','"
    stri3 = current_time.strftime("%Y-%m-%d %H:%M:%S")
    stri4 = "')"
    cursor.execute(stri1+x+stri2+stri3+stri4)
    str1 = "with parkinglot as ( "
    str2 = "select * from park_info natural join park_location natural join park_name natural join park_info_realtime ) "
    str3 = "select park_name , park_location , distance , business_hour , weekday_cost, holiday_cost, totalspace_car, totalspace_mot, freespace_car, freespace_mot, update_time "
    str4 = "from (location_with_parkinglot natural join location_table) inner join parkinglot "
    str5 = "on parkinglot.park_id = location_with_parkinglot.park_id "
    str6 = "where location_name = '" 
    str7 = "'    and freespace_car > 0 "
    str8 = " order by distance limit 5 "
    if z == "motor":
        str7 = ("'    and freespace_mot > 0 ")
    elif z == "car":
        str7 = ("'    and freespace_car > 0 ")
    else:
        str7 = ("'    and freespace_car > 0 ")
    print(str1 + str2 + str3 + str4 + str5 + str6 + x + str7 + str8)
    cursor.execute(str1 + str2 + str3 + str4 + str5 + str6 + x + str7 + str8)
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
    