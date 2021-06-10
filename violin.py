import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from csv_to_pandas import csv_to_pandas
import pandas as pd
def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

def violins_chromatic_v_diatonic(partition, data,x,y,name=""):
    n_groups = len(data.groupby(partition))
    fig, axes = plt.subplots(ncols=n_groups, sharex=True, sharey=True)
    fig.suptitle(name, fontsize=16)

    for ax, (n,grp) in zip(axes, data.groupby(partition)):
        # sns.violinplot(x=x, y=y, data=grp, ax=ax, palette=['green','purple'], inner="stick")
        # sns.violinplot(x=x, y=y, data=grp, ax=ax, inner="box")

        sns.barplot(x=x, y=y, data=grp, ax=ax, palette=['#A64669', '#9688F2'])
        sns.stripplot(x=x, y=y, data=grp, palette=['black'], ax=ax)

        # sns.pointplot(x=x, y=y, hue="subject", data=grp,
        #               palette=["black"], ax=ax)
        ax.set_title(n)
    for ax in axes:
        try:
            ax.get_legend().remove()
        except:
            1+1
    # plt.ylim(0.35, 0.9)
    # plt.ylim(800, 2750)
    plt.savefig(name + ".svg")
    plt.show()

def bars_contour_v_pitch(partition, data,x,y,name=""):
    n_groups = len(data.groupby(partition))
    fig, axes = plt.subplots(ncols=n_groups, sharex=True, sharey=True,figsize=[3,8])
    fig.suptitle(name, fontsize=16)
    for ax, (n,grp) in zip(axes, data.groupby(partition)):
        # print(grp.head())
        sns.barplot(x=x, y=y, data=grp,ax=ax)
        sns.stripplot(x=x, y=y, data=grp,palette=['black'],ax=ax)
        # change_width(ax, .35)

        ax.set_title(n)
    for i,ax in enumerate(axes):
        try:
            if(i>0): ax.set_ylabel('')
            ax.set_xlabel('')
            ax.set(xticklabels=[''])
            ax.get_legend().remove()
        except:
            1+1
    plt.show()



format_general = [
    {
        'static': {
            'type':'chromatic',
            'manipulation':'contour'
        },
        'dynamic':{
            'subject':'Subject_ID',
            'bias': '%_chromatic_swapped',
            'rt': 'rt_chromatic_swapped',
            'rt_type': 'rt_chromatic',
            'pc_decoy':'%_correct_decoys',
            'shifted-swapped':'c_shifted-swapped',
            'pc_decoy_at_condition': '%_correct_chromatic_same_v_swapped',
            'decoy_rt_type':'rt_chromatic_decoy',
            'decoy_rt_at_condition': 'rt_chromatic_decoy_swapped'

        }
    },
    {
        'static': {
            'type':'chromatic',
            'manipulation':'pitch'
        },
        'dynamic':{
            'subject':'Subject_ID',
            'bias': '%_chromatic_shifted',
            'rt': 'rt_chromatic_shifted',
            'rt_type': 'rt_chromatic',
            'pc_decoy':'%_correct_decoys',
            'shifted-swapped':'c_shifted-swapped',
            'pc_decoy_at_condition': '%_correct_chromatic_same_v_shifted',
            'decoy_rt_type':'rt_chromatic_decoy',
            'decoy_rt_at_condition': 'rt_chromatic_decoy_shifted'
        }
    },
    {
        'static': {
            'type':'diatonic',
            'manipulation':'contour'
        },
        'dynamic':{
            'subject':'Subject_ID',
            'bias': '%_diatonic_swapped',
            'rt': 'rt_diatonic_swapped',
            'rt_type': 'rt_diatonic',
            'pc_decoy':'%_correct_decoys',
            'shifted-swapped':'d_shifted-swapped',
            'pc_decoy_at_condition': '%_correct_diatonic_same_v_swapped',
            'decoy_rt_type': 'rt_diatonic_decoy',
            'decoy_rt_at_condition': 'rt_diatonic_decoy_swapped'
        }
    },
    {
        'static': {
            'type':'diatonic',
            'manipulation':'pitch',
        },
        'dynamic':{
            'subject':'Subject_ID',
            'bias': '%_diatonic_shifted',
            'rt': 'rt_diatonic_shifted',
            'rt_type': 'rt_diatonic',
            'pc_decoy':'%_correct_decoys',
            'shifted-swapped':'d_shifted-swapped',
            'pc_decoy_at_condition': '%_correct_diatonic_same_v_shifted',
            'decoy_rt_type': 'rt_diatonic_decoy',
            'decoy_rt_at_condition': 'rt_diatonic_decoy_shifted'
        }
    }
]

#8 notes
data_8 = csv_to_pandas("./8-note.csv",format_general)
data_8.head()
data_8["bias"] = data_8['bias'].astype('float')
data_8["pc_decoy_at_condition"] = data_8['pc_decoy_at_condition'].astype('float')
data_8["rt"] = data_8['rt'].astype('float')
data_8["rt_type"] = data_8['rt_type'].astype('float')
data_8["shifted-swapped"] = data_8['shifted-swapped'].astype('float')


# violins_chromatic_v_diatonic('type',data_8,x='manipulation',y='bias',name="8-notes-bias")
# violins_chromatic_v_diatonic('type',data_8,x='manipulation',y='pc_decoy_at_condition',name="8-notes-decoy")
# violins_chromatic_v_diatonic('type',data_8, x='manipulation',y='rt',name='RT_8-notes')
# violins_chromatic_v_diatonic('type',data_8, x='type',y='rt_type',name='RT_8-notes_ALL')
# violins_chromatic_v_diatonic('type',data_8, x='type',y='shifted-swapped',name='8-notes')

#12 notes
data_12 = csv_to_pandas("./12-note.csv",format_general)
data_12["bias"] = data_12['bias'].astype('float')
data_12["pc_decoy_at_condition"] = data_12['pc_decoy_at_condition'].astype('float')
data_12["rt"] = data_12['rt'].astype('float')
data_12["rt_type"] = data_12['rt_type'].astype('float')
data_12["shifted-swapped"] = data_12['shifted-swapped'].astype('float')


# violins_chromatic_v_diatonic('type',data_12,x='manipulation',y='bias',name="12-notes-bias")
# violins_chromatic_v_diatonic('type',data_12,x='manipulation',y='pc_decoy_at_condition',name="12-notes-decoy")
# violins_chromatic_v_diatonic('type',data_12, x='manipulation',y='rt',name='RT_12-notes')
# violins_chromatic_v_diatonic('type',data_12, x='type',y='rt_type',name='RT_12-notes_ALL')
# violins_chromatic_v_diatonic('type',data_12, x='type',y='shifted-swapped',name='12-notes')



#16 notes
data_16 = csv_to_pandas("./16-note.csv",format_general)
data_16["bias"] = data_16['bias'].astype('float')
data_16["pc_decoy_at_condition"] = data_16['pc_decoy_at_condition'].astype('float')
data_16["rt"] = data_16['rt'].astype('float')
data_16["rt_type"] = data_16['rt_type'].astype('float')
data_16["shifted-swapped"] = data_16['shifted-swapped'].astype('float')

# violins_chromatic_v_diatonic('type',data_16,x='manipulation',y='bias',name="16-notes-bias")
# violins_chromatic_v_diatonic('type',data_16,x='manipulation',y='pc_decoy_at_condition',name="16-notes-decoy")
# violins_chromatic_v_diatonic('type',data_16, x='manipulation',y='rt',name='RT_16-notes')
# violins_chromatic_v_diatonic('type',data_16, x='type',y='rt_type',name='RT_16-notes_ALL')
# violins_chromatic_v_diatonic('type',data_16, x='type',y='shifted-swapped',name='16-notes')


#12 note decoys
data_12d = csv_to_pandas("./12-note-decoys.csv",format_general)
data_12d["pc_decoy_at_condition"] = data_12d['pc_decoy_at_condition'].astype('float')
data_12d["decoy_rt_type"] = data_12d['decoy_rt_type'].astype('float')
data_12d["decoy_rt_at_condition"] = data_12d['decoy_rt_at_condition'].astype('float')

violins_chromatic_v_diatonic('type',data_12d,x='manipulation',y='pc_decoy_at_condition',name="12-notes-decoy")
# violins_chromatic_v_diatonic('type',data_12d, x='type',y='decoy_rt_type',name='12Decoy_RT_Type')
# violins_chromatic_v_diatonic('type',data_12d, x='manipulation',y='decoy_rt_at_condition',name='12Decoy_RT_at_Cond')



#16 note decoys
data_16d = csv_to_pandas("./16-note-decoys.csv",format_general)
data_16d["pc_decoy_at_condition"] = data_16d['pc_decoy_at_condition'].astype('float')
data_16d["decoy_rt_type"] = data_16d['decoy_rt_type'].astype('float')
data_16d["decoy_rt_at_condition"] = data_16d['decoy_rt_at_condition'].astype('float')

# violins_chromatic_v_diatonic('type',data_16d,x='manipulation',y='pc_decoy_at_condition',name="16-notes-decoy")
# violins_chromatic_v_diatonic('type',data_16d, x='type',y='decoy_rt_type',name='16Decoy_RT_Type')
# violins_chromatic_v_diatonic('type',data_16d, x='manipulation',y='decoy_rt_at_condition',name='16Decoy_RT_at_Cond')



#12 notes below chance
data_12b = csv_to_pandas("./12-note-BC.csv",format_general)
data_12b["bias"] = data_12b['bias'].astype('float')
data_12b["pc_decoy_at_condition"] = data_12b['pc_decoy_at_condition'].astype('float')
data_12b["rt"] = data_12b['rt'].astype('float')
data_12b["rt_type"] = data_12b['rt_type'].astype('float')
data_12b["shifted-swapped"] = data_12b['shifted-swapped'].astype('float')

# violins_chromatic_v_diatonic('type',data_12b,x='manipulation',y='bias',name="B12-notes-bias")
# violins_chromatic_v_diatonic('type',data_12b,x='manipulation',y='pc_decoy_at_condition',name="B12-notes-decoy")
# violins_chromatic_v_diatonic('type',data_12b, x='manipulation',y='rt',name='BRT_12-notes')
# violins_chromatic_v_diatonic('type',data_12b, x='type',y='rt_type',name='BRT_12-notes_ALL')
# violins_chromatic_v_diatonic('type',data_12b, x='type',y='shifted-swapped',name='B12-notes')

#16 notes below chance
data_16b = csv_to_pandas("./16-note-BC.csv",format_general)
data_16b["bias"] = data_16b['bias'].astype('float')
data_16b["pc_decoy_at_condition"] = data_16b['pc_decoy_at_condition'].astype('float')
data_16b["rt"] = data_16b['rt'].astype('float')
data_16b["rt_type"] = data_16b['rt_type'].astype('float')
data_16b["shifted-swapped"] = data_16b['shifted-swapped'].astype('float')

# violins_chromatic_v_diatonic('type',data_16b,x='manipulation',y='bias',name="B16-notes-bias")
# violins_chromatic_v_diatonic('type',data_16b,x='manipulation',y='pc_decoy_at_condition',name="B16-notes-decoy")
# violins_chromatic_v_diatonic('type',data_16b, x='manipulation',y='rt',name='BRT_16-notes')
# violins_chromatic_v_diatonic('type',data_16b, x='type',y='rt_type',name='BRT_16-notes_ALL')
# violins_chromatic_v_diatonic('type',data_16b, x='type',y='shifted-swapped',name='B16-notes')


format_66 = []

for i in range(66):
    obj = {
        'static': {
            'set':i,
        },
        'dynamic':{
            'value':str(i),
        }
    }
    format_66.append(obj)


# score_matrix = csv_to_pandas("./score-matrix.csv",format_66)
# nan_value = float("NaN")
# score_matrix.replace("", nan_value, inplace=True)
# score_matrix.dropna(inplace=True)
# score_matrix["value"] = score_matrix["value"].astype('float')
# plt.figure(figsize=(10,5))
# # sns.violinplot(x="set", y="value", data=score_matrix)
# ax = sns.barplot(x="set", y="value", data=score_matrix, palette=['green', 'purple'])
# # sns.stripplot(x="set", y="value", data=score_matrix, palette=['black'], ax=ax)
# plt.show()


# score_matrix = csv_to_pandas("./score-matrix.csv",format_66)
# score_matrix = csv_to_pandas("./RT-matrix.csv",format_66)
# score_matrix = score_matrix.loc[score_matrix['set'].isin(['3','65'])]
# score_matrix = score_matrix.loc[score_matrix['set'].isin(['11','30'])]
# score_matrix = score_matrix.loc[score_matrix['set'].isin(['5','43'])]


# nan_value = float("NaN")
# score_matrix.replace("", nan_value, inplace=True)
# score_matrix.dropna(inplace=True)
#
# score_matrix["value"] = score_matrix["value"].astype('float')
# plt.figure(figsize=(10,5))
# # # ax = sns.violinplot(x="set", y="value", data=score_matrix)
# # # sns.violinplot(x="set", y="value", data=score_matrix, ax=ax, inner="box")
# # ax = sns.barplot(x="set", y="value", data=score_matrix, palette=['green', 'purple'])
# ax = sns.regplot(x="set", y="value", data=score_matrix,
#                  x_estimator=np.mean)
# # sns.stripplot(x="set", y="value", data=score_matrix, palette=['black'], ax=ax)
#
# plt.savefig("test.svg")
# plt.show()

format_sem = [
    {
        'static': {
            'type':'score',
        },
        'dynamic':{
            'value':'Score SEM',
        }
    },
{
        'static': {
            'type':'pos',
        },
        'dynamic':{
            'value':'Pos SEM'
        }
    },
]

# sem_matrix = csv_to_pandas("./SEM.csv",format_sem)
# nan_value = float("NaN")
# sem_matrix.replace("", nan_value, inplace=True)
# sem_matrix.dropna(inplace=True)
# sem_matrix["value"] = sem_matrix["value"].astype('float')
# # plt.figure(figsize=(100,5))
# # sns.violinplot(x="set", y="value", data=score_matrix)
# ax = sns.barplot(x="type", y="value", data=sem_matrix, palette=['green', 'purple'])
# # sns.stripplot(x="type", y="value", data=sem_matrix, palette=['black'], ax=ax)
# plt.ylim(0.3, 0.4)
#
# plt.show()

# DvND = pd.read_csv('./DvND_Score.csv')
# ax = sns.barplot(x="Type", y="Score", data=DvND, palette=['green', 'purple'])
#
# plt.show()
#
#
# DvND = pd.read_csv('./ScoreByP5.csv')
# ax = sns.barplot(x="Type", y="Score", data=DvND, palette=['green', 'purple','red'])
# sns.stripplot(x="Type", y="Score", data=DvND, palette=['black'], ax=ax)
# plt.show()
#
#
# DScoreByP5 = pd.read_csv('./DScoreByP5.csv')
# ax = sns.barplot(x="Type", y="Score", data=DScoreByP5, palette=['green', 'purple','red'])
# sns.stripplot(x="Type", y="Score", data=DScoreByP5, palette=['black'], ax=ax)
# ax.set_title("Diatonic")
# plt.show()
#
# NDScoreByP5 = pd.read_csv('./NDScoreByP5.csv')
# ax = sns.barplot(x="Type", y="Score", data=NDScoreByP5, palette=['green', 'purple','red'])
# sns.stripplot(x="Type", y="Score", data=NDScoreByP5, palette=['black'], ax=ax)
# ax.set_title("NDiatonic")
# plt.show()