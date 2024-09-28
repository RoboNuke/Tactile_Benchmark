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

@dataclass
class Args:
    wandb_entity: str = "hur"
    """the entity (team) of wandb's project"""
    wandb_project_name: str = "In-Contact_Baseline"
    """the wandb's project name"""
    exp_prefex: Optional[str] = "Test_Run"
    """the name of this experiment"""
    local_save_path: str = 'data/plots/test'

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
    subname: Optional[str] = '2024-09-25_18:46'
    """subname to filter in data_by_subname mode"""


def prepMetadata(run):
    metadata = {}
    metadata['tags'] = run.tags
    metadata['config'] = run.config
    return metadata

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




