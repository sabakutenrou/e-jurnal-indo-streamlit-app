import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

# from streamlit_gallery.utils.model import load_data

# ------ Tokenizing ---------
#remove number
def remove_number(text):
    return  re.sub(r"\d+", "", text)

#remove punctuation
def remove_punctuation(text):
    return text.translate(str.maketrans("","",string.punctuation))

#remove whitespace leading & trailing
def remove_whitespace_LT(text):
    return text.strip()

#remove multiple whitespace into single whitespace
def remove_whitespace_multiple(text):
    return re.sub('\s+',' ',text)

# remove single char
def remove_singl_char(text):
    return re.sub(r"\b[a-zA-Z]\b", "", text)

# NLTK tokenisasi kata 
def word_tokenize_wrapper(text):
    return word_tokenize(text)


# ALL PROCESSES TOGETHER
def preprocess(text):
    # remove tab, new line, ans back slice
    text = text.replace('\\t'," ").replace('\\n'," ").replace('\\u'," ").replace('\\',"")
    # remove non ASCII (emoticon, chinese word, .etc)
    text = text.encode('ascii', 'replace').decode('ascii')
    # remove mention, link, hashtag
    text = ' '.join(re.sub("([@#][A-Za-z0-9]+)|(\w+:\/\/\S+)"," ", text).split())
    # remove incomplete URL
    text = text.replace("http://", " ").replace("https://", " ")
    text = re.sub(r"\d+", "", text) # remove_number(text)
    text = text.translate(str.maketrans("","",string.punctuation)) # remove_punctuation(text)
    text = text.strip() # remove_whitespace_LT(text)
    text = re.sub('\s+',' ',text) # remove_whitespace_multiple(text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text) # remove_singl_char(text)
    # text = word_tokenize_wrapper(text)
    return text

# DATA = load_data()
# DATA['abstrak_tokens'] = DATA['abstrak'].apply(preprocess)

# print('Hasil tokenisasi : \n') 
# print(DATA['abstrak_tokens'].head())
# print('\n\n\n')