import sqlite3
import os
import pickle
import fnmatch

project_dir=os.path.abspath('/home/k1801311/Documents/KCL/PhD/codes/pattern_walker')
data_path=os.path.join(project_dir,'outputs/20201009')
db_path=os.path.join(project_dir,'outputs/db_test.db')


def locate(pattern, root_path):
    for path, dirs, files in os.walk(root_path):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)


def get_data():
    for file in locate('*.pkl',data_path):
        with open(file,'rb') as f:
            data=[]
            while 1:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
            for time in data[0]:
                yield ( data[2]['job_id'],data[2]['seed'],data[2]['branching_factor'],data[2]['gamma'],data[2]['string_len'],time )



def db_connect(db_path,db_name):
    """ create a database connection to a SQLite database """
    conn = None

    try:
        conn = sqlite3.connect(db_path+db_name)
    except sqlite3.Error as e:
        if not path.exists(db_path):
            print ("Invalid directory!\n" )

        print(e)

    return conn

def db_check_table_existence(conn):
    c=conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS data (
        ID integer PRIMARY KEY not null,
        m integer,
        mu real,
        D integer,
        N integer,
        Q real
        );''')
    conn.commit()

    return

###############################################################################
###############################################################################
if not os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    with conn:
        cur =conn.cursor()
        cur.execute("DROP TABLE IF EXISTS times")
        cur.execute("CREATE TABLE times (id INT, seed INT, branching_factor FLOAT,gamma FLOAT, string_len INT, hitting_time INT )")
        for datum in get_data():
            cur.execute("INSERT INTO times VALUES(?,?,?,?,?,?)",datum)
    conn.commit()
