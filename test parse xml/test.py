import xml.etree.ElementTree as ET
import streamlit as st

file_path ="173.cermxml"
tree = ET.parse(file_path)

# get the parent tag 
root = tree.getroot()

# judul
for elm in root.findall("./front/article-meta/title-group/article-title"):          # for elm in root.findall("./body/sec[@id='sec-1']/p"):
    st.write()                                                              #     st.write(elm.text)
    st.write()                                                                      # not this as abstract is like this too

# judul
list_judul = [elm.text for elm in root.findall("./front/article-meta/title-group/article-title")]

# tahun
list_tahun = [elm1.text for elm1 in root.findall("./front/article-meta/pub-date/year")]

# abstrak
list_abstract = [elm.text for elm in root.findall("./front/article-meta/abstract/")]

# author
list_author = [elm.text for elm in root.findall("./front/article-meta/contrib-group/contrib[@contrib-type='author]")]

# nama jurnal
list_nama_jurnal = [elm.text for elm in root.findall("./front/journal-meta/ournal-title-group/journal-title")]

# keyword
list_kwd = [elm.text for elm in root.findall("./front/article-meta/kwd-group/kwd")]
# if english wipe out, go:
# crawl that less than 10 WORDS !!, if yes:
#  then use other scenario (normal pdf text read)

from abstrak import pdfparser
import re
file_path = file_path.replace('cermxml', 'pdf')
parsed = pdfparser(file_path)
pattern = re.compile('(?<=abstrak)(.*)(?=kata kunci)', re.IGNORECASE)
abstract = re.findall(pattern, parsed)
