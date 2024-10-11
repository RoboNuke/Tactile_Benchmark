from data.data_manager import DataManager
from collections import defaultdict
import os
import random
import time
from dataclasses import dataclass, field
from typing import Optional, List
import tyro
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

@dataclass
class Args:
    wandb_entity: str = "hur"
    """the entity (team) of wandb's project"""
    wandb_project_name: str = "Tester"#"In-Contact_Baseline"
    """the wandb's project name"""
    exp_prefex: Optional[str] = "Test_Run"
    """the name of this experiment"""
    local_save_path: str = 'data/plots/oct_10_exps/base_60M'

    data_to_plot: List = field(default_factory=lambda :['/success_rate', 
                               '/reward', 
                               '/failure_rate', 
                               '/episode_length',
                               '/returns',
                               '_smoothness/sum_squared_velocity',
                               '_smoothness/avg_max_force'
                            ])
    """Which values in run history to plot"""
    data_y_labels: List = field(default_factory= lambda :['Success Rate (%)',
                                'Avg Reward per Step',
                                'Failure Rate (%)',
                                'Episode Length (steps)',
                                'Avg Total Episode Reward',
                                'Sum Squared Velocity ( (m/s)^2 )',
                                'Avg Max Force (N)'
                            ])
    """Name for the y-axis of plots in data_to_plot"""
    subplot_titles: List = field(
        default_factory=lambda :['Success Rate vs Step',
                                'Avg Step Reward vs Step',
                                'Failure Rate vs Step',
                                'Episode Length vs Step',
                                'Avg Total Episode Reward vs Step',
                                'Sum Squared Velocity vs Step',
                                'Avg Max Force vs Step'
                            ])
    """Plot titles of plots in data_to_plot"""
    eval_plots: bool = True
    """Create plots for evaluation data"""

    data_by_subname: bool = True
    """plot only data with this subname"""
    subname: Optional[str] = 'FPiH_peg_dmg_500.0_1_2024-10-10_15:59'
    """subname to filter in data_by_subname mode"""

    data_by_config_key: bool = False
    """Key to sort data by"""
    config_key: Optional[str] = 'exp_max_dmg_force'
    """Config key to sort data by if data_by_config_key is set to true"""

    data_by_h_config_key: bool = False

def prepMetadata(run):
    metadata = {}
    metadata['tags'] = run.tags
    metadata['config'] = run.config
    return metadata

def make_dict_from_lists(lists):
    """
    Create a dictionary from n lists where each 
    list represents the keys at a specific level.
    """
    result = {}
    def add_keys(dic, new_keys, end=False):
        for new_key in new_keys:
            if end:
                dic[new_key] = []
            else:
                dic[new_key] = {}
        return dic            

    last_dics = []
    last_list = []
    for key_list in lists:
        end = key_list == lists[-1]
        if len(result.keys()) == 0:
            result = add_keys(result, key_list)
            last_dics = result.values()
            last_list = key_list
        else:
            for last_dic in last_dics:
                add_keys(last_dic, key_list, end)
            if not end:
                last_dics = [val for dic in last_dics for val in dic.values()]
                last_list = key_list
    return result

if __name__=="__main__":
    args = tyro.cli(Args)
    colors = ['r', 'g', 'b', 'y', 'b']

    assert len(args.data_to_plot) == len(args.data_y_labels), 'Not enough y labels'
    assert len(args.data_to_plot) == len(args.subplot_titles), 'Not enough subplot titles'

    DM = DataManager(
        args.wandb_project_name, 
        args.wandb_entity
    )

    # make folders
    data_sources = ['train']
    if args.eval_plots:
        data_sources.append("eval")

    for data_source in data_sources:
        Path(f'{args.local_save_path}/{data_source}/').mkdir(parents=True, exist_ok=True)

    if args.data_by_subname:
        # single plot for each data_to_plot (+1 for eval)
            
        runs = DM.download_runs_by_subname(args.subname)
        assert len(runs) > 0, f"No runs with subname {args.subname}"
        metadata = prepMetadata(runs[0])
        for data_source in data_sources:
            for data_idx, data_name in enumerate(args.data_to_plot):
                fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
                ax.set_title(args.subplot_titles[data_idx])
                ax.set_ylabel(args.data_y_labels[data_idx])
                ax.set_xlabel('Steps')

                DM.add_runs_to_plot(runs, ax, f'{data_source}{data_name}', 'b')

                name = data_name.split("/")[-1]

                fig.savefig(f'{args.local_save_path}/{data_source}/{name}.png')
        plt.show()

    if args.data_by_config_key:
        for data_source in data_sources:
            for data_idx, data_name in enumerate(args.data_to_plot):
                name = data_name.split("/")[-1]
                DM.plot_runs_with_key(
                    args.config_key, 
                    var_name=f'{data_source}{data_name}', 
                    title=args.subplot_titles[data_idx],
                    xlab='Steps',
                    ylab=args.data_y_labels[data_idx],
                    save_path=f'{args.local_save_path}/{data_source}/{name}.png',
                )

    

    if args.data_by_h_config_key:
        colors=[(1,0,0),(0,1,0),(0,0,1)]
        key_hs = ['include_force', 'exp_max_dmg_force']
        key_vs = [[True, False], [100, 250, 100000]]
        groups = make_dict_from_lists(key_vs)
        runs = DM.api.runs(DM.entity + "/" + DM.project,
                           filters={"tags": {"$in":["present_oct_10"]}})

        for run in runs:
            lis = groups
            skip = False
            for key_h in key_hs:
                try:
                    lis = lis[run.config[key_h]]
                except:
                    skip = True
            if not skip:
                lis.append(run)
        
        #for key in groups.keys():
        #    for key2 in groups[key].keys():
        #        print(len(groups[key][key2]))
        
        for data_source in data_sources:
            for data_idx, data_name in enumerate(args.data_to_plot):
                print(f'data name:{data_name}')
                ylims = None
                if '_smoothness/avg_max_force' in data_name:
                    ylims = [0.0, 250.0]
                title=args.subplot_titles[data_idx]
                xlab='Steps'
                ylab=args.data_y_labels[data_idx]

                # create plot
                fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
                ax.set_title(title)
                ax.set_ylabel(ylab)
                ax.set_xlabel(xlab)

                if type(ylims) == list:
                    ax.set_ylim(ylims[0], ylims[1])

                # convert to numpy lists
                for k,group_name in enumerate(groups.keys()):
                    for j, subgroup_name in enumerate([100, 250, 100000]):
                        print(group_name, subgroup_name)
                        ys = []
                        step = None
                        n_steps = 10000000
                        for run in groups[group_name][subgroup_name]:
                            his = run.history()
                            #print(his.keys())
                            step = his['_step'].to_numpy()
                            y = his[f'{data_source}{data_name}'].to_numpy()
                            step = step[~np.isnan(y)]
                            n_steps = min(n_steps, len(step))
                            y = y[~np.isnan(y)]
                            ys.append(y)
                        ys = [y[:n_steps] for y in ys]
                        step = step[:n_steps]
                        DM.plot_with_ci(ax, 
                                    step, 
                                    np.array(ys),
                                    data_name=f'{group_name}:{subgroup_name}',
                                    color=[color * (1.0 if group_name else 0.5) for color in colors[j]]
                        )
                ax.set_xlim((0,step[-1]))
                # save file
                set_of_lines = ax.get_lines()
                print(len(set_of_lines))
                new_lines = []
                for i in range(len(set_of_lines)):
                    if i % 2 == 0:
                        new_lines.append(set_of_lines[i])

                plt.legend(new_lines, groups[True].keys())
                name = data_name.split("/")[-1]
                plt.savefig(f'{args.local_save_path}/{data_source}/{name}.png')




