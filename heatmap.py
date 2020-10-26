import sqlite3 as sql
import pandas as pd
import numpy as np
from itertools import product
import seaborn as sns
import matplotlib.pyplot as plt

#N=15
#o_range=[np.round(x*1/N,2) for x in range(0,N+2,2)]
o_range=np.round(np.arange(0.05,1.05,0.05),2)
g_range=np.round(np.arange(0.05,1.05,0.05),2)
N_range=[15]
param_range=product(o_range,g_range)
mfpts={ N : pd.DataFrame(np.zeros((len(g_range),len(o_range))),index=g_range,columns=o_range) for N in N_range}
stds={ N : pd.DataFrame(np.zeros((len(g_range),len(o_range))),index=g_range,columns=o_range) for N in N_range}

for N in N_range:
    mfpts[N].columns.name='overlap'
    mfpts[N].index.name='gamma'
    stds[N].columns.name='overlap'
    stds[N].index.name='gamma'

    with sql.connect('../fpts.db') as conn:
        cur=conn.cursor()

        for (o,g) in param_range:
            cur.execute("SELECT hitting_time FROM redraw_patterns WHERE string_len = ? AND overlap = ? AND gamma = ?",(N,o,g))
            temp=list(cur.fetchall())
            mfpts[N][o][g]=np.mean(temp)
            stds[N][o][g]=np.std(temp)
            print(N,o,g,mfpts[N][o][g],stds[N][o][g])

    print(mfpts[N])
    print(stds[N])
    mfpts[N].to_csv('mfpts_N_{}.csv'.format(N))
    stds[N].to_csv('stds_{}.csv'.format(N))
    plt.figure()
    sns.heatmap(mfpts[N])
    plt.savefig('mfpts_heatmap_N_{}.pdf'.format(N))
    plt.close()
    plt.figure()
    sns.heatmap(stds[N])
    plt.savefig('stds_N_{}.pdf'.format(N))
    plt.close()
