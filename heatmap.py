import sqlite3 as sql
import argparse
import pandas as pd
import numpy as np
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt
import multiprocessing as mp

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

parser.add_argument("--param_table", default='unknown', dest="param_table",
    help="List of distinct values for string length, overlap and gamma available. Will search table under --table for distinc triples of none is given or table does not exist."
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

parser.add_argument("--num_cores", default=1, type=int, dest="num_cores",
    help="Length of patterns to take into account. (default: %(default)s)"
)

args=parser.parse_args()

def table_exists(db_cursor,name):
    query = "SELECT 1 FROM sqlite_master WHERE type='table' and name = ?"
    return db_cursor.execute(query, (name,)).fetchone() is not None

def sort_df(df):
    df.columns=np.round(df.columns,args.round)
    df.index=np.round(df.index,args.round)
    df=df.reindex(np.sort(df.index))
    df=df.reindex(columns=np.sort(df.columns))
    return df

with sql.connect(args.database) as conn:
    if table_exists(conn,args.table_name):
        conn.row_factory = lambda cursor, row: row[0]
        overlap_cur=conn.cursor()
        gamma_cur=conn.cursor()
        overlap_range=[]
        gamma_range=[]
        param_range=[]
        if table_exists(conn,args.param_table):
            param_table=args.param_table
        else:
            param_table=args.table_name

        overlap_range=overlap_cur.execute("SELECT DISTINCT overlap FROM {} WHERE string_len=?".format(param_table),(args.string_len,)).fetchall()
        gamma_range=gamma_cur.execute("SELECT DISTINCT gamma FROM {} WHERE string_len=?".format(param_table),(args.string_len,)).fetchall()
        param_range=product(overlap_range,gamma_range)
        print('Parameters found: ', gamma_range,overlap_range)
        mfpts=pd.DataFrame(np.zeros((len(gamma_range),len(overlap_range))),index=gamma_range,columns=overlap_range)
        stds=pd.DataFrame(np.zeros((len(gamma_range),len(overlap_range))),index=gamma_range,columns=overlap_range)

        mfpts.columns.name='overlap'
        mfpts.index.name='gamma'
        stds.columns.name='overlap'
        stds.index.name='gamma'
        N=args.string_len
        query="SELECT hitting_time FROM {} WHERE overlap = ? AND gamma = ? and string_len=?".format(args.table_name)
        def get_times(params):
            cur=conn.cursor()
            print(params)
            temp=cur.execute(query,(params[0],params[1],N)).fetchall()
            mean=np.mean(temp)
            std=np.std(temp)
            print(N,params[0],params[1],mean,std)
            return (params[0],params[1],mean,std)

        if args.num_cores>1:
            print("multiprocessing.")
            with mp.Pool(args.num_cores) as p:
                for (o,g,mean,std) in p.map(get_times,param_range):
                    mfpts[o][g]=mean
                    stds[o][g]=std
        else:
            print("singleprocessing")
            for (o,g) in param_range:
                cur=conn.cursor()
                cur.execute("SELECT hitting_time FROM {} WHERE overlap = ? AND gamma = ? and string_len=?".format(args.table_name),(o,g,N))
                temp=list(cur.fetchall())
                mfpts[o][g]=np.mean(temp)
                stds[o][g]=np.std(temp)
                print(N,',',o,',',g,',',mfpts[o][g],',',stds[o][g])

        mfpts=sort_df(mfpts)
        stds=sort_df(stds)
        mfpts.columns.name='overlap'
        mfpts.index.name='gamma'
        stds.columns.name='overlap'
        stds.index.name='gamma'

        print(mfpts)
        print(stds)
        mfpts.to_csv('{}_mfpts_N_{}.csv'.format(args.output,N))
        stds.to_csv('{}_stds_N_{}.csv'.format(args.output,N))
        plt.figure()
        sns.heatmap(mfpts)
        plt.tight_layout()
        plt.savefig('{}_mfpts_heatmap_N_{}.pdf'.format(args.output,N))
        plt.close()
        plt.figure()
        sns.heatmap(stds)
        plt.tight_layout()
        plt.savefig('{}_stds_heatmap_N_{}.pdf'.format(args.output,N))
        plt.close()
