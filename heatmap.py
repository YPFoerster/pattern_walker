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
    help="Output_prefix, including path, if appropriate. (default: %(default)s)"
    )

parser.add_argument("--round", default=2, type=int ,dest="round",
    help="Digit to round gamma and overlap to in the output csv's. (default: %(default)s)"
    )

parser.add_argument("--string_len", default=15, type=int, dest="string_len",
    help="Length of patterns to take into account. (default: %(default)s)"
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
        overlap_range=overlap_cur.execute("SELECT DISTINCT overlap FROM {} WHERE string_len=? order by overlap ASC".format(args.table_name),(args.string_len,)).fetchall()
        gamma_range=overlap_cur.execute("SELECT DISTINCT gamma FROM {} WHERE string_len=? order by gamma ASC".format(args.table_name),(args.string_len,)).fetchall()
        param_range=product(overlap_range,gamma_range)
        print(overlap_range,gamma_range,param_range)
        N_range=[args.string_len]
        mfpts={ N : pd.DataFrame(np.zeros((len(gamma_range),len(overlap_range))),index=gamma_range,columns=overlap_range) for N in N_range}
        stds={ N : pd.DataFrame(np.zeros((len(gamma_range),len(overlap_range))),index=gamma_range,columns=overlap_range) for N in N_range}
        for N in N_range:
            mfpts[N].columns.name='overlap'
            mfpts[N].index.name='gamma'
            stds[N].columns.name='overlap'
            stds[N].index.name='gamma'

            for (o,g) in param_range:
                cur=conn.cursor()
                cur.execute("SELECT hitting_time FROM redraw_patterns WHERE overlap = ? AND gamma = ? and string_len=?",(o,g,N))
                temp=list(cur.fetchall())
                mfpts[N][o][g]=np.mean(temp)
                stds[N][o][g]=np.std(temp)
                print(N,o,g,mfpts[N][o][g],stds[N][o][g])

            mfpts[N].columns=np.round(mfpts[N].columns,args.round)
            mfpts[N].index=np.round(mfpts[N].index,args.round)
            stds[N].columns=np.round(stds[N].columns,args.round)
            stds[N].index=np.round(stds[N].index,args.round)
            print(mfpts[N])
            print(stds[N])
            mfpts[N].to_csv('{}_mfpts_N_{}.csv'.format(args.output,N))
            stds[N].to_csv('{}_stds_N_{}.csv'.format(args.output,N))
            plt.figure()
            sns.heatmap(mfpts[N])
            plt.tight_layout()
            plt.savefig('{}_mfpts_heatmap_N_{}.pdf'.format(args.output,N))
            plt.close()
            plt.figure()
            sns.heatmap(stds[N])
            plt.tight_layout()
            plt.savefig('{}_stds_heatmap_N_{}.pdf'.format(args.output,N))
            plt.close()
