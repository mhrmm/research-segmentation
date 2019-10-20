import unittest
from readdata import read_from_training_data

class TestLemmas(unittest.TestCase):

    def test_sample_sense_pairs(self):
        train_data = ['a', 't', ' ', 't', 'h', 'e', ' ', 'z', 'o', 'o', '。', '\n']
        sents, flags = read_from_training_data(train_data)
        assert sents == [['a', 't', 't', 'h', 'e', 'z', 'o', 'o', '。']]
        assert flags == [[False, True, False, False, True, False, False, False, False]]




if __name__ == "__main__":
	unittest.main()
