# compare 3 runs with baseline data
from os import listdir
import pandas as pd
from os.path import isfile, join


# take name and iterate over seed

def create_summary(dfs):
    out = []
    for df in dfs:
        before_45M = df[df["frames"] <= 45000000]
        ret_mean = before_45M['return_mean'].mean()
        ret_max = before_45M['return_mean'].max()
        quality = before_45M['rud_quality'].mean()

        around_03_ret = df[df["return_mean"] >= 0.3]
        frames_min = around_03_ret['frames'].min() / 1000000
        out.append([ret_mean, ret_max, frames_min,quality])

    out = pd.DataFrame(out, columns=['ret_mean', 'ret_max', 'frame_min','quality'])
    return out.describe().transpose()[['mean', 'std']].round(2)


def df_to_plus_minus(df):
    # mystr = " " + u'\u00b1' + " "
    mystr = " mypm "

    return df[df.columns].apply(lambda x: mystr.join(x.dropna().astype(str)), axis=1)


def create_df_slice(data):
    sum_df = create_summary(data)
    return df_to_plus_minus(sum_df)


directory = "../scripts/logs/masterThesis/rudVBert/"
baseline_name = "rudbase"

dirs = [f for f in listdir(directory) if f != '.DS_Store']

baseline_data = []
to_compared = []
for d in dirs:
    data = pd.read_csv(directory + d + "/log.csv")[['frames', 'return_mean','rud_quality']]
    if baseline_name.lower() in d.lower():
        baseline_data.append(data)
    else:
        to_compared.append(data)

assert len(baseline_data) == len(to_compared) == 3

base_df = create_df_slice(baseline_data).rename('rudder vanilla')
rud_df = create_df_slice(to_compared).rename('rudder bert')

final_df = pd.concat([base_df, rud_df], axis=1)
latex_str = final_df.to_latex()
latex_str = latex_str.replace('mypm', '$ \pm $')
print(latex_str)
print('d')

#               baseline | run x
# mean ret      3
# max ret       3
# frames min    233
