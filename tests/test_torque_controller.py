import unittest
from tasks.for_env import *
from tasks.fragile_insert import *
from learning.agent import *
from wrappers.n_force_wrapper import NForceWrapper

from tests.dummy_env import *

class TestNForce(unittest.TestCase):
    #@classmethod
    #def setUpClass(cls):
    #    pass

    #def setUp(self):
    #    print("Problem setup!")

    #def tearDown(self):
    #    pass
    def test_action_space(self):
        # ensures actions space is correct
        assert(1==1)

    def test_single_action_space(self):
        # ensures single action space correct
        assert(1==1)

    def test_set_drive_property(self):
        # ensures properties update for each joint
        assert(1==1)

    def test_reset(self):
        assert(1==1)

    def test_partial_reset(self):
        assert(1==1)

    def test_set_action(self):
        assert(1==1)

    def test_get_set_state(self):
        # checks that state can be get/set
        assert(1==1)

    def test_set_drive_targets(self):
        # ensures each joint torque set 
        assert(1==1)

    def test_before_sim_step(self):
        assert(1==1)