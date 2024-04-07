import configparser
import os
import unittest
import pandas as pd
import sys

sys.path.insert(1, os.path.join(os.getcwd(), "src"))

from train import Train


class TestTrain(unittest.TestCase):

    def setUp(self) -> None:
        self._Train = Train(params=None)

    def test_train(self):
        self.assertEqual(self._Train.train_model(), True)


if __name__ == "__main__":
    unittest.main()