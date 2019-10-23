import unittest
from readdata import read_segmented_string, split_long_string, read_train_data
from readdata import read_test_data
from readdata import read_from_training_data, read_from_testing_data

class TestReadData(unittest.TestCase):

    def test_read_segmented_string(self):
        line = '在  半梦半醒  之间  。'
        sent, flags = read_segmented_string(line)
        assert sent == ['在', '半', '梦', '半', '醒', '之', '间', '。']
        assert flags == [True, False, False, False, True, False, True, True]

    def test_split_long_string(self):
        line = '在 半梦半醒 之间 。缓缓 游动 于 瓷 身 。'
        sents = split_long_string(line, 13)
        assert len(sents) == 2
        assert sents[0] == '在 半梦半醒 之间 。'
        assert sents[1] == '缓缓 游动 于 瓷 身 。'
        sents = split_long_string(line, 24)
        assert len(sents) == 1
        assert sents[0] == line
        sents = split_long_string(line, 6)
        assert len(sents) == 5
        assert sents[0] == '在 半梦半醒'
        assert sents[1] == '之间 。'
        assert sents[2] == '缓缓 游动'
        assert sents[3] == '于 瓷 身'
        assert sents[4] == '。'

    def test_read_train_data(self):
        line = '在  半梦半醒  之间  。 缓缓  游动  于  瓷  身  。'
        sents, flagsets = read_train_data([line], limit=300)
        assert sents == [['在', '半', '梦', '半', '醒', '之', '间', '。', '缓', 
                          '缓', '游', '动', '于', '瓷', '身', '。']]
        assert flagsets == [[True, False, False, False, True, False, True, 
                             True, False, True, False, True, True, True, 
                             True, True]]
        sents, flagsets = read_train_data([line], limit=20)
        assert sents == [['在', '半', '梦', '半', '醒', '之', '间', '。'], 
                         ['缓', '缓', '游', '动', '于', '瓷', '身', '。']]
        assert flagsets == [[True, False, False, False, True, False, True, True], 
                            [False, True, False, True, True, True, True, True]]
        
    def test_read_test_data(self):
        lines = ['在半梦半醒之间。','缓缓游动于瓷身。烘托一种氛围。']
        sents = read_test_data(lines, limit=10)
        assert len(sents) == 2
        assert sents[0] == ['在半梦半醒之间。']
        assert sents[1] == ['缓缓游动于瓷身。', '烘托一种氛围。']        
        sents = read_test_data(lines, limit=40)
        assert len(sents) == 2
        assert sents[0] == ['在半梦半醒之间。']
        assert sents[1] == ['缓缓游动于瓷身。烘托一种氛围。']        
     
    def test_read_from_training_data(self):
        train_data = ['a', 't', ' ', 't', 'h', 'e', ' ', 'z', 'o', 'o', '。', '\n']
        sents, flags = read_from_training_data(train_data)
        assert sents == [['a', 't', 't', 'h', 'e', 'z', 'o', 'o', '。']]
        assert flags == [[False, True, False, False, True, False, False, False, False]]

    def test_read_from_testing_data(self):
        char_string = '共同创造美好的新世纪——二○○一年新年贺词\n（二○○○年十二月三十一日）（附图片1张）\n女士们，先生们，同志们，朋友们：\n'
        sents = read_from_testing_data(char_string)
        
        assert sents[0] == ['共', '同', '创', '造', '美', '好', '的', '新', 
                    '世', '纪', '—', '—', '二', '○', '○', '一', '年', '新', 
                    '年', '贺', '词']
        assert sents[1] == ['（', '二', '○', '○', '○', '年', '十', '二', '月', 
                    '三', '十', '一', '日', '）', '（', '附', '图', '片', '1', 
                    '张', '）']
        assert sents[2] == ['女', '士', '们', '，', '先', '生', '们', '，', 
                    '同', '志', '们', '，', '朋', '友', '们', '：']

        

if __name__ == "__main__":
	unittest.main()
