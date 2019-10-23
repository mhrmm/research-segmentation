import unittest
import os
from segmenter import segment_sent, segment_sents, segment_file
from segmenter import BMES, XE

class TestSegmenter(unittest.TestCase):

    def model(self, sent):
        if sent == '在半梦半醒之间。':
            return [True, False, False, False, True, False, True, True]
        elif sent == '缓缓游动于瓷身。':
            return [False, False, False, True, True, False, True, True]
        else:
            return [True, False, True, False, False, True, True]
    
    def test_segment_sent(self):
        line = '在半梦半醒之间。'
        result = segment_sent(self.model, line)
        assert result == '在  半梦半醒  之间  。'
        
    def test_segment_sents(self):
        lines = [['在半梦半醒之间。'], ['缓缓游动于瓷身。', '烘托一种氛围。']]
        result = segment_sents(self.model, lines)                
        assert result == '在  半梦半醒  之间  。\n缓缓游动  于  瓷身  。  烘  托一  种氛围  。'

    def test_segment_file(self):
        segment_file(self.model, 'testfiles/unsegmented.utf8', 'foo.utf8')        
        with open('foo.utf8') as inhandle:
            lines = [line.strip() for line in inhandle]
        os.remove('foo.utf8')
        assert len(lines) == 2
        assert lines[0] == '在  半梦半醒  之间  。'
        assert lines[1] == '缓缓游动  于  瓷身  。  烘  托一  种氛围  。'

    def test_BMES_encoding(self):
        line = '在  半梦半醒  之间  。'
        encoding = BMES()
        tokens, flags = encoding.encode(line)        
        assert tokens == ['在', '半', '梦', '半', '醒', '之', '间', '。']
        assert flags == [BMES.S, BMES.B, BMES.M, BMES.M, BMES.E, BMES.B, BMES.E, BMES.S]
        
    def test_BMES_decoding(self):
        tokens = ['在', '半', '梦', '半', '醒', '之', '间', '。']
        flags = [BMES.S, BMES.B, BMES.M, BMES.M, BMES.E, BMES.B, BMES.E, BMES.S]        
        encoding = BMES()
        result = encoding.decode(tokens, flags)        
        assert result == '在  半梦半醒  之间  。'
  
        

if __name__ == "__main__":
	unittest.main()
