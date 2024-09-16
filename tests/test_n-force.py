import unittest
from tasks.wiping import *
from tasks.barnicle_scrap import *
from learning.agent import *

class TestNForce(unittest.TestCase):
    #@classmethod
    #def setUpClass(cls):
    #    pass

    #def setUp(self):
    #    print("Problem setup!")

    #def tearDown(self):
    #    pass

    def test_barn_ns(self):
        # barnicle scrap with different ns
        # checks
        # - shape of obs
        # - init values (before n steps)
        # - stored in ['extra']['force']
        assert(1==0)

    def test_wiping_ns(self):
        assert(1==0)

    def test_ppih_ns(self):
        assert(1==0)

    def test_vector_state(self):
        # ensures can go from dict to vector
        assert(1==0)