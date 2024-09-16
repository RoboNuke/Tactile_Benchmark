import wandb
from dataclasses import dataclass
import tyro


class DataManager():
    def __init__(self):
        pass

    def login_wandb(self):
        """
            Logs in to wandb
        """
        pass
    
    def init_new_run(self,config):
        """
            Start a new run with tags in config
        """
        pass

    def add_scalar(self, tag, scalar_value, step):
        pass

    def add_ckpt(self, ckpt):
        """
            Save model parameters
        """
        pass

    def finish(self):
        """
            Correctly shutdown wandb
        """
        pass

    def download_run_data(self):
        """
            Get all run data and sort into dataframe
        """
        pass

    def plot_with_ci(self, ax, x, y, 
                     title="Cool Title", 
                     xLab="Step", yLab="Func",
                     data_name="data"):
        """
            Calculates the 95% CI for dataset y
            and plots it on plot ax
        """
        pass

    def group_runs_by_key(self, key):
        """
            Given a key, group all run data
            into each unique value of key
        """
        pass

    def plot_runs_with_key(self, key, save_path=""):
        """
            Sorts data into groups given by key, and 
            then plots each group with 95% CI then
            saves the figure to save_path as vector img
        """
        pass




    


