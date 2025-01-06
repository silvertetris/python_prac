from transformers import BertTokenizer
from unicodedata import normalize

vocab_url = 'https://raw.githubusercontent.com/snunlp/KR-BERT/master/krbert_pytorch/pretrained/vocab_snu_subchar12367.txt'

tokenizer_krbert = BertTokenizer.from_pretrained(vocab_url, do_lower_case=True)

# convert a string into sub-char
def to_subchar(string):
    return normalize('NFKD', string)

sentence = '토크나이저 예시입니다.'
print(tokenizer_krbert.tokenize(to_subchar(sentence)))
