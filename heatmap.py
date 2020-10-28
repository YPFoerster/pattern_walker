import sqlite3 as sql
import argparse
import pandas as pd
import numpy as np
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="""
    Read the pickled results of a simulation (first list being meta data,
    second list hitting times) in a given directory and insert them in a
    given database.
    """)


parser.add_argument("--database", default='unknown', dest='database',
    help="Database to insert data into. (default: %(default)s)"
    )

parser.add_argument("--table", default='unknown', dest='table_name',
    help="Name of the table in database. (default: %(default)s)"
    )

parser.add_argument("--output", default='unknown', dest='output',
    help="output_prefix, including path, if appropriate. (default: %(default)s)"
    )

args=parser.parse_args()

def table_exists(db_cursor,name):
    query = "SELECT 1 FROM sqlite_master WHERE type='table' and name = ?"
    return db_cursor.execute(query, (name,)).fetchone() is not None


with sql.connect(args.database) as conn:
    if table_exists(conn,args.table_name):
        conn.row_factory = lambda cursor, row: row[0]
        overlap_cur=conn.cursor()
        gamma_cur=conn.cursor()
        overlap_range=overlap_cur.execute("SELECT DISTINCT overlap FROM {} order by overlap ASC".format(args.table_name)).fetchall()
        gamma_range=overlap_cur.execute("SELECT DISTINCT gamma FROM {} order by gamma ASC".format(args.table_name)).fetchall()
        param_range=product(overlap_range,gamma_range)
        print(overlap_range,gamma_range,param_range)
        N_range=[50]
        mfpts={ N : pd.DataFrame(np.zeros((len(gamma_range),len(overlap_range))),index=gamma_range,columns=overlap_range) for N in N_range}
        stds={ N : pd.DataFrame(np.zeros((len(gamma_range),len(overlap_range))),index=gamma_range,columns=overlap_range) for N in N_range}
        for N in N_range:
            mfpts[N].columns.name='overlap'
            mfpts[N].index.name='gamma'
            stds[N].columns.name='overlap'
            stds[N].index.name='gamma'

            for (o,g) in param_range:
                cur=conn.cursor()
                cur.execute("SELECT expected_FPT FROM redraw_patterns WHERE overlap = ? AND gamma = ? and string_len=?",(o,g,N))
                temp=list(cur.fetchall())
                mfpts[N][o][g]=np.mean(temp)
                stds[N][o][g]=np.std(temp)
                print(N,o,g,mfpts[N][o][g],stds[N][o][g])

            print(mfpts[N])
            print(stds[N])
            mfpts[N].to_csv('{}_mfpts_N_{}.csv'.format(args.output,N))
            stds[N].to_csv('{}_stds_{}.csv'.format(args.output,N))
            f=plt.figure()
            f.tight_layout()
            sns.heatmap(mfpts[N])
            plt.savefig('{}_mfpts_heatmap_N_{}.pdf'.format(args.output,N))
            plt.close()
            plt.figure()
            sns.heatmap(stds[N])
            plt.savefig('{}_stds_N_{}.pdf'.format(args.output,N))
            plt.close()
