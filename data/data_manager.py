import wandb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

class DataManager():
    def __init__(self, 
                 project = "Tester", 
                 entity= "hur",
                 api_key = 'a593b534585893ad93cf62243228a866c9053247'
        ):
        self.project = project
        self.entity = entity
        self.key= api_key
        self.ckpt_enabled = False
        self.run_name = ''
        self.login_wandb()
        self.api = wandb.Api()

    def __del__():
        print("deleted")

    def login_wandb(self):
        """
            Logs in to wandb
        """
        wandb.login(
            key=self.key,
            force=True
        )
    
    def init_new_run(self, 
                     run_name, 
                     config,
                     tags = []
        ):
        """
            Start a new run with tags in config
        """
        wandb.init(entity=self.entity,
                   project=self.project,
                   name=run_name,
                   config=config,
                   tags = tags)
        self.run_name = run_name

    def add_scalar(self, 
                   data, 
                   step=None, 
                   commit=False
        ):
        if step is not None:
            commit = False
        wandb.log(data, step=step, commit=commit)

    def add_gif(self, 
                data_or_path, 
                tag = 'gif',
                step=None, 
                cap="my gif", 
                fps=10, 
                commit=True
        ):
        wandb.log({
            tag:wandb.Video(
                        data_or_path=data_or_path,
                        caption=cap,
                        fps=fps,
                        format='gif'
                    )
            }, step=step, commit=commit)
        
    def add_mp4(self, 
                data_or_path, 
                tag = 'gif',
                step=None, 
                cap="my mp4", 
                fps=10
        ):
        wandb.log({
            tag:wandb.Video(
                        data_or_path=data_or_path,
                        caption=cap,
                        fps=fps,
                        format='mp4'
                    )
            }, step=step)

    def add_ckpt(self, ckpt_path, wandb_path):
        """
            Save model parameters
            - ckpt_path: path on local files
            - wandb path: location to save it to
        """
        # set up a ckpt folder and artifact logger
        #self.ckpt_art = wandb.Artifact(name=f"{self.run_name}_ckpts",
        #                                type='ckpt')
        #self.ckpt_art.add_file(local_path= ckpt_path,
        #                       name=name)
        #self.ckpt_art.save()
        
    def get_dir(self):
        return wandb.run.dir
    
    def add_save(self, con, base_path=None):
        print(con)
        if base_path == None:
            wandb.save(con)
        else:
            wandb.save(con, base_path=base_path)

    def finish(self, quiet=True, exit_code=0):
        """
            Correctly shutdown wandb
        """
        wandb.finish(exit_code=exit_code, quiet=quiet)

    def download_run_data(self, run_id):
        """
            Get all run data and sort into dataframe
        """
        ru = self.api.run(path=f'{self.entity}/{self.project}/{run_id}')
        return ru

    def download_all_run_data(self, save_path="tests/tot_data.csv"):
        runs = self.api.runs(self.entity + "/" + self.project)
        df = None
        for k, run in enumerate(runs):
            his = run.history()
            if k  == 0:
                df = his
            else:
                df = pd.concat([df, his])
            
            
        df.to_csv(save_path, index=False)

    def download_run_data_by_name(self, run_name):
        runs = self.api.runs(self.entity + "/" + self.project)

        for run in runs:
            if run.name == run_name:
                return run
            
        return None

    def group_runs_by_key(self, key):
        """
            Given a key, group all run data
            into each unique value of key
        """
        runs = self.api.runs(self.entity + "/" + self.project)
        groups = {}
        for run in runs:
            val = run.config[key]
            if val in groups:
                groups[val].append(run)
            else:
                groups[val] = [run]
            
        return groups
        
        

    def plot_runs_with_key(self, key, var_name="loss", 
                           title="Force Encoding's effect on Loss",
                           xlab='Steps',
                           ylab='Loss',
                           save_path="",
                           colors = ['r', 'g', 'b', 'y', 'b']):
        """
            Sorts data into groups given by key, and 
            then plots each group with 95% CI then
            saves the figure to save_path as vector img
        """
        # get groups
        groups = self.group_runs_by_key(key)

        # create plot
        fig, ax = plt.subplots(figsize=(10, 5), dpi=200)
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel(xlab)
        # convert to numpy lists
        max_step = -1
        n_steps = 10000000
        for k,group_name in enumerate(groups):
            ys = []
            step = None
            for run in groups[group_name]:
                his = run.history()
                step = his['_step'].to_numpy()
                max_step = max(max_step, step[-1])
                n_steps = min(n_steps, len(step))
                y = his[var_name].to_numpy()
                ys.append(y)
            self.plot_with_ci(ax, 
                        step, 
                        np.array(ys),
                        data_name=group_name,
                        color=colors[k]
            )
        ax.set_xlim((0,max_step))
        # save file
        plt.legend()
        plt.savefig(save_path)

    def plot_with_ci(self, ax, x, y, 
                     color='b',
                     data_name="data"):
        """
            Calculates the 95% CI for dataset y
            and plots it on plot ax
        """
        y_mean = np.mean(y.T, axis=1)
        y_std = np.std(y.T, axis=1)
        # Calculate the confidence interval (e.g., 95%)
        ci = 1.96 * y_std / np.sqrt(len(x))

        # Plot the data and the confidence interval
        ax.plot(x, y_mean, 
                f'{color}-', 
                label=data_name)
        ax.fill_between(x, 
                        y_mean - ci, 
                        y_mean + ci, 
                        color=color, 
                        alpha=0.2)
        




    


