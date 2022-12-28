import string 
import re #regex library

# import word_tokenize & FreqDist from NLTK
from nltk.tokenize import word_tokenize 
from nltk.probability import FreqDist

from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory, StopWordRemover, ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.tokenize.treebank import TreebankWordDetokenizer
from nltk.corpus import stopwords

# from streamlit_gallery.utils.model import load_data

# ------ Tokenizing ---------
#remove number
# def remove_number(text):
#     return  re.sub(r"\d+", "", text)

# #remove punctuation
# def remove_punctuation(text):
#     return text.translate(str.maketrans("","",string.punctuation))

# #remove whitespace leading & trailing
# def remove_whitespace_LT(text):
#     return text.strip()

# #remove multiple whitespace into single whitespace
# def remove_whitespace_multiple(text):
#     return re.sub('\s+',' ',text)

# # remove single char
# def remove_singl_char(text):
#     return re.sub(r"\b[a-zA-Z]\b", "", text)

# # NLTK tokenisasi kata 
# def word_tokenize_wrapper(text):
#     return word_tokenize(text)

prefix = ['meng','per','ber','ter','di','ke','se','peng']
suffix = ['kan','an','i']
infix = ['el','er','em']

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
    text = text.lower()
    return text

# def remove_stopwords(kalimat):
#     import string
#     # import nltk
#     # nltk.download('punkt')
#     # ambil stopword bawaan
#     stop_factory = StopWordRemoverFactory().get_stop_words()
#     more_stopword = ['daring', 'online', 'mau']

#     # kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online. Menurut Andi belanja online lebih praktis & murah."
#     kalimat = kalimat.translate(str.maketrans('','',string.punctuation)).lower()

#     # menggabungkan stopword
#     data = stop_factory + more_stopword
    
#     dictionary = ArrayDictionary(data)
#     text = StopWordRemover(dictionary)
#     tokens = word_tokenize(text.remove(kalimat))
    
#     # print(tokens)

#     return text.remove(kalimat)
f = open("stopwords.txt", "r")
stopword_list = []
for line in f:
    stripped_line = line.strip()
    line_list = stripped_line.split()
    stopword_list.append(line_list[0])
f.close()

def remove_stopwordss(text):
    

    len(stopword_list)

    # text = "Apa yang dimakan oleh bambang apakah mengapa".lower()
    text = text.lower()
    text_tokens = word_tokenize(text)

    
    tokens_without_sw = [word for word in text_tokens if not word in stopword_list]
    tokens_without_sw_eng = [token for token in tokens_without_sw if token not in stopwords.words("english")]
    # print("After stopwords removed")
    # print(tokens_without_sw)
    return TreebankWordDetokenizer().detokenize(tokens_without_sw_eng)


factory = StemmerFactory()
stemmer = factory.create_stemmer()
def stemming(kalimat):
    
    
    # kalimat = "Andi kerap melakukan transaksi rutin secara daring atau online. Menurut Andi belanja online lebih praktis & murah."

    hasil = stemmer.stem(kalimat)
    return hasil
# DATA = load_data()
# DATA['abstrak_tokens'] = DATA['abstrak'].apply(preprocess)

# print('Hasil tokenisasi : \n') 
# print(DATA['abstrak_tokens'].head())
# print('\n\n\n')

import pandas as pd

def readkamus(namafile):
    text_file = open(namafile, "r")
    lines = text_file.readlines()
    lines2 = []
    for i in lines:
        i = i.replace('\n', '')
        lines2.append(i)
    return lines2

def cekkamus(kata,kamus):
    for i in kamus:
        if i == kata:
            return True
    return False

def pemisahan_prefix(kata,prefix):
    for i in prefix:
        if kata.startswith(i):
            kata = kata[len(i):len(kata)]
            return kata ,i
    return kata, ''

def pemisahan_infix(kata,infix):
    for i in infix:
        if i in kata:
            kata = kata.replace(i,"")
            return kata,i
    return kata, ''

def pemisahan_suffix(kata, suffix):
    for i in suffix:
        if kata.endswith(i):
            kata = kata[0:(len(kata)-len(i))]
            return kata,i
    return kata, ''
def penemuan_kata_dasar(kata,kamus):
    kata = kata.lower()
    simpankatatidakbaku = kata
    ada = cekkamus(kata,kamus)
    if not ada:
        kata, imbuhan = pemisahan_prefix(kata, prefix)
        #print('imbuhannya adalah :', imbuhan)
    else:
        return kata
    ada = cekkamus(kata, kamus)
    if not ada:
        kata,akhiran = pemisahan_suffix(kata,suffix)
        #print('akhirannya adalah :',akhiran)
    else:
        return kata
    ada = cekkamus(kata, kamus)
    if not ada:
        kata, tengahan = pemisahan_infix(kata,infix)
        #print('tengahannya adalah :',tengahan)
    else:
        return kata
    ada = cekkamus(kata, kamus)
    if not ada:
        #print('kata tidak baku')
        return simpankatatidakbaku
    return kata



# prefix = ['meng','per','ber','ter','di','ke','se','peng']
# suffix = ['kan','an','i']
# infix = ['el','er','em']
# kamus = readkamus('kata-dasar.txt')
# '''
# kata  = input()
# kata = penemuan_kata_dasar(kata,kamus)
# print('kata bakunya adalah :',kata)
# '''
# kalimat = input().lower()
# listkalimat = [str(i) for i in kalimat.split()]
# kalimatdasar = []
# strkalimatdasar = ''
# for i in listkalimat:
#     i = penemuan_kata_dasar(i,kamus)
#     kalimatdasar.append(i)
#     strkalimatdasar = strkalimatdasar + ' ' + i
# print(strkalimatdasar)

df = pd.read_csv('dataset.csv', encoding='utf-8')

# import nltk
# # nltk.download('punkt')
# nltk.download('stopwords')

df['judul-jurnal'] = df['judul-jurnal'].apply(preprocess)
df['judul-jurnal'] = df['judul-jurnal'].apply(remove_stopwordss)
df['judul-jurnal'] = df['judul-jurnal'].apply(stemming)

df['abstrak-jurnal'] = df['abstrak-jurnal'].apply(preprocess)
df['abstrak-jurnal'] = df['abstrak-jurnal'].apply(remove_stopwordss)
df['abstrak-jurnal'] = df['abstrak-jurnal'].apply(stemming)

df.to_csv('dataset-clean.csv')
