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
    str1 = 'select location_name ,count(location_name) as counting '
    str2 = 'from history_search '
    str3 = 'group by location_name '
    str4 = 'order by counting desc '
    str5 = 'limit 5 '
    cursor.execute(str1 + str2 + str3 + str4 + str5)
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
    