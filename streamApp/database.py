# Import libraries.
import sqlite3




# Create a database.
conn = sqlite3.connect('registered.db')
c = conn.cursor()
# Create a table in the db
c.execute(
    '''CREATE TABLE IF NOT EXISTS attendancee (Date text, student_name text, attendance text, arrival_time text)''')

sql1 = 'DELETE FROM attendancee'
c.execute(sql1)

conn.commit()
conn.close()
