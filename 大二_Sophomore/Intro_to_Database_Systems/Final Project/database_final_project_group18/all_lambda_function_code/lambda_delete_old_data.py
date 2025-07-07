import urllib.request as request
import json
import psycopg2 
import sys
import datetime

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
    current_time =datetime.datetime.now()
    weekago = current_time - datetime.timedelta(days = 7)
    temp = weekago.strftime("%Y-%m-%d %H:%M:%S")
    print(temp)
    cursor.execute("delete from park_info_realtime_old where update_time < '" + temp + "'")
    cursor.close()
    conn.close()
    return {
        "statuscode": 200,
    }