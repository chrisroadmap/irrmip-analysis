# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from scipy.signal import savgol_filter

# %%
os.makedirs('../plots/', exist_ok=True)

pl.style.use('../defaults.mplstyle')

# %%
models = ['CESM2', 'CESM2-ground', 'CNRM', 'NorESM', 'E3SM']

# %%
df = {}
for model in models:
    df[model] = pd.read_csv(f'../output/{model}.csv', index_col=0, header=[0,1])

# %%
df['CNRM'].rename(
    columns={
        "irr_r1i1p1f2": "IRR_01", 
        "irr_r2i1p1f2": "IRR_02", 
        "irr_r3i1p1f2": "IRR_03", 
        "irr_r4i1p1f2": "IRR_04", 
        "irr_r5i1p1f2": "IRR_05", 
        "noirr_r1i1p1f2": "NOI_01", 
        "noirr_r2i1p1f2": "NOI_02", 
        "noirr_r3i1p1f2": "NOI_03", 
        "noirr_r4i1p1f2": "NOI_04", 
        "noirr_r5i1p1f2": "NOI_05", 
    }, inplace=True
)

# %%
runids = {}
for model in models:
    runids[model] = []
    for expt in df[model].columns.get_level_values('experiment').unique():
        runids[model].append(expt[-2:])
    runids[model] = list(set(runids[model]))
runids

# %%
runids

# %%
erf = {}
for model in runids:
    erf[model] = {}
    if model=='CNRM':
        sw = 'rsut'
        lw = 'rlut'
    else:
        sw = 'rsnt'
        lw = 'rlnt'
    for runid in runids[model]:
        print(model, runid)
        erf[model][runid] = (
            (df[model][f'IRR_{runid}'][sw] - df[model][f'IRR_{runid}'][lw]) - 
            (df[model][f'NOI_{runid}'][sw] - df[model][f'NOI_{runid}'][lw])
        )

# %%
colors = {
    'CESM2': 'purple',
    'CESM2-ground': 'red',
    'CNRM': 'green',
    'NorESM': 'blue',
    'E3SM': 'orange'
}

# %%
for model in runids:
    for runid in runids[model]:
        pl.plot(erf[model][runid], color=colors[model])

# %% [markdown]
# ## remove obvious outliers (found by looking at CSV)

# %%
for runid in runids['CESM2-ground']:
    pl.plot(erf['CESM2-ground'][runid], label=runid)
pl.legend()

# %%
# 1914 is probably fine, since 1913 is OK we'll leave 1914 in
erf['CESM2-ground']['01'].loc[1901:1920]

# %%
erf['CESM2-ground']['01'].loc[1901:1912] = np.nan

# %%
for runid in runids['E3SM']:
    pl.plot(erf['E3SM'][runid], label=runid)
pl.legend()

# %%
erf['E3SM']['01'].loc[2014] = np.nan

# %%
for model in runids:
    for runid in runids[model]:
        pl.plot(erf[model][runid], color=colors[model])

# %%
# find the ensemble mean
for model in runids:
    df_mean = pd.DataFrame()
    for runid in runids[model]:
        df_mean = pd.concat((df_mean, erf[model][runid].rename(runid)), axis=1)
    erf[model]['mean'] = df_mean.mean(axis=1)

# %%
for model in runids:
    pl.plot(erf[model]['mean'], color=colors[model])

# %%
# create an 11-year smoothing filter
for model in models:
    pl.plot(
        erf[model]['mean'].index,
        savgol_filter(np.concatenate((np.zeros(10), erf[model]['mean'], erf[model]['mean'].values[-1]*np.ones(10))), 11, 1)[10:-10],
        color=colors[model]
    )

# %%
# create a 21-year smoothing filter
for model in models:
    pl.plot(
        erf[model]['mean'].index,
        savgol_filter(np.concatenate((np.zeros(20), erf[model]['mean'], erf[model]['mean'].values[-1]*np.ones(20))), 21, 1)[20:-20],
        color=colors[model]
    )

# %%
for model in runids:
    print(model, erf[model]['mean'].mean(), erf[model]['mean'].std())

# %%
