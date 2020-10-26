# compare 3 runs with baseline data
from os import listdir
import pandas as pd
from os.path import isfile, join

# take name and iterate over seed

def create_summary(dfs):
    out=[]
    for df in dfs:
        before_45M = df[df["frames"]<=45000000]
        ret_mean = before_45M['return_mean'].mean()
        ret_max = before_45M['return_mean'].max()

        around_03_ret = df[df["return_mean"]>=0.3]
        frames_min = around_03_ret['frames'].min()
        out.append([ret_mean,ret_max,frames_min])

    out = pd.DataFrame(out, columns=['ret_mean','ret_max','frame_min'])
    return out






directory = "../scripts/logs/masterThesis/baselineVSStandard/"

dirs = [f for f in listdir(directory)]

baseline_name = "norud"
baseline_data = []
to_compared = []
for d in dirs:
    data = pd.read_csv(directory + d + "/log.csv")[['frames', 'return_mean']]
    if baseline_name in d.lower():
        baseline_data.append(data)
    else:
        to_compared.append(data)

assert len(baseline_data) == len(to_compared) == 3
create_summary(baseline_data)