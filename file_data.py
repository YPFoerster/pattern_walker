import sqlite3 as sql
import pickle as pickle
import argparse
import os
import fnmatch
import numpy as np

parser = argparse.ArgumentParser(description="""
    Read the pickled results of a simulation (first list being meta data,
    second list hitting times) in a given directory and insert them in a
    given database.
    """)

parser.add_argument("--location", default='unknown', dest='location',
    help="Directory that contains data in .pkl files. (default: %(default)s)"
    )

parser.add_argument("--database", default='unknown', dest='database',
    help="Database to insert data into. (default: %(default)s)"
    )

parser.add_argument("--table", default='unknown', dest='table_name',
    help="Name of the table in database. (default: %(default)s)"
    )

args=parser.parse_args()

def table_exists(db_cursor,name):
    query = "SELECT 1 FROM sqlite_master WHERE type='table' and name = ?"
    return db_cursor.execute(query, (name,)).fetchone() is not None

def locate(pattern, root_path):
    for path, dirs, files in os.walk(root_path):
        for filename in fnmatch.filter(files, pattern):
            yield os.path.join(path, filename)

def get_data_from(path):
    for file in locate('*.pkl',path):
        print(file)
        with open(file,'rb') as f:
            data=[]
            while 1:
                try:
                    data.append(pickle.load(f))
                except EOFError:
                    break
            try:
                for time in data[1]:
                    yield ( data[0].get('job_id'),file,data[0].get('seed'),data[0].get('branching_factor'),data[0].get('gamma'),data[0].get('string_len'),data[0].get('duplicate_patterns'),data[0].get('overlap'),data[0].get('target_node'),np.real(data[0].get('mfpt')),data[0].get('job_dir'),time )
            except IndexError as e:
                print('Error while accessing ',file)
                print(e)
                continue

with sql.connect(args.database) as conn:
    cur=conn.cursor()

    if table_exists(cur,args.table_name):
        #cur.execute("DROP TABLE IF EXISTS redraw_patterns")
        #cur.execute("CREATE TABLE IF NOT EXISTS {} (id INT,location TEXT ,seed INT, branching_factor FLOAT,gamma FLOAT, string_len INT,number_duplicates INT,overlap FLOAT,target_node TEXT, expected_FPT FLOAT, job_dir TEXT, hitting_time INT)".format(args.table_name))
        for datum in get_data_from(args.location):
            cur.execute("INSERT INTO {} VALUES(?,?,?,?,?,?,?,?,?,?,?,?)".format(args.table_name),datum)
    else:
        print('Table does not exist.')
