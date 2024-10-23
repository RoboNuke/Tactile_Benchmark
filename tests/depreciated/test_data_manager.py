import unittest
from tasks.wiping import *
from tasks.barnicle_scrap import *
from learning.agent import *
from data.data_manager import DataManager
import wandb

class TestDataManager(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.DM = DataManager()

    #def setUp(self):
    #    print("Problem setup!")

    #def tearDown(self):
    #    pass

    def test_inits(self):
        # test login
        self.DM.login_wandb()
        self.init_new_run()
        self.add_data(scale=1.0)
        self.init_new_run_diff_config()
        self.add_data(scale=0.5)
        self.init_new_run_diff_config(config = {
                "observation_space": "state",
                "force_encoder": "N_step",
                "policy_type": "NN",
                "action_space": "hybrid"
            }, name='third_run')
        self.add_data(scale=2.0)
        self.DM.finish(exit_code=-1)
        assert(wandb.run is None)

    def init_new_run(self):
        assert(wandb.run is None)
        config = {
            "observation_space": "state",
            "force_encoder": "None",
            "policy_type": "NN",
            "action_space": "ee_delta"
        }

        self.DM.init_new_run('first_run', config)
        con = wandb.config
        for key in config:
            assert(con[key] == config[key], f'Configuration key {key} is not equal')

    def init_new_run_diff_config(self, config=None, name='second_run'):
        if config is None:
            config = {
                "observation_space": "state",
                "force_encoder": "1_Step",
                "policy_type": "NN",
                "action_space": "ee_delta"
            }
        self.DM.finish()
        assert(wandb.run is None)
        self.DM.init_new_run(name, config)
        assert(wandb.run is not None, "New run not initialized correctly")
        con = wandb.config
        for key in config:
            assert(con[key] == config[key], f'New Config key {key} is not equal')

    def add_data(self, scale):
        self.DM.add_scalar({'loss':1.0*scale, 'acc':0.25*scale}, step=0)
        self.add_ckpt('ckpt_0')
        self.DM.add_scalar({'loss':0.5*scale}, commit=False, step=10)
        self.DM.add_scalar({'acc':0.5*scale}, commit=False, step=10)
        self.DM.add_gif(data_or_path="tests/Animated-GIF-Banana.gif",
                        tag='eval/gifs',
                        step=10,
                        cap="Everything is Bananas!",
                        fps=10,
                        commit=True
        )
        self.add_ckpt('ckpt_1')
        
    def add_ckpt(self, name='ckpt_1'):
        self.DM.add_ckpt("tests/model_test.ckpt", name)

    def test_download(self):
        run_name = "first_run"
        df = self.DM.download_run_data('xn80wn9p')
        print(df.name)
        print(df.config.items())
        print(df.history().head)

    def test_download2(self):
        run_name = "first_run"
        df = self.DM.download_run_data_by_name(run_name)
        print(df.name)
        print(df.config.items())
        print(df.history().head)

    def test_group_runs(self):
        group_list = self.DM.group_runs_by_key('force_encoder')
        assert(len(group_list) == 3)
        for group in group_list:
            for run in group_list[group]:
                print(run)
                print(run.history())
        group_list2 = self.DM.group_runs_by_key('action_space')
        assert(len(group_list2) == 2)
        assert(not group_list2 == group_list)

    def test_plots(self):
        import random
        make_runs = False
        # create a bunch of random runs
        if make_runs:
            forces = ["None", "1_Step", "N_Step"]
            config = {
                "observation_space": "state",
                "force_encoder": None,
                "policy_type": "NN",
                "action_space": "ee_delta"
            }

            for k, force_type in enumerate(forces):
                config['force_encoder'] = force_type
                for i in range(5):
                    self.DM.init_new_run(f'{force_type}_{i}', config)
                    for i in range(100):
                        self.DM.add_scalar({'loss':float(k) + random.random() - 0.5})
                    self.DM.finish() 
        
        self.DM.plot_runs_with_key(key='force_encoder', 
                                   var_name='loss',
                                   save_path='tests/test_fig.png')
        self.DM.download_all_run_data() 