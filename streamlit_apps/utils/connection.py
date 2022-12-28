import streamlit as st
import mysql.connector

import pymysql
import sqlalchemy

# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    return mysql.connector.connect(**st.secrets["mysql"], charset='utf8')

conn = init_connection()

username= "root"
host = "localhost"
port = 3306
database = "e_jurnal_db"
user = "root"
password = "12345"

# conn = sqlalchemy.create_engine("mysql+pymysql://" + username + ":" + password + "@" + host + "/" + database)
# df.to_sql('dataset', con = conn, if_exists = 'replace',index = False
#     # , chunksize = 1000
#     )

# Perform query.
    # Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

rows = run_query("SELECT * FROM e_jurnal_db.dataset;")

# Print results.
# for row in rows:
#     st.write(f"{row[0]} has a :{row[1]}:")