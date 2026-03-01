import sqlite3
import pandas as pd

conn = sqlite3.connect('crowd_data.db')
df = pd.read_sql_query("SELECT * FROM crowd_log", conn)
print(df)
conn.close()
