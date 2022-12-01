import xml.etree.ElementTree as ET
import streamlit as st

file_path ="173.cermxml"
tree = ET.parse(file_path)

# get the parent tag 
root = tree.getroot()

# judul
for elm in root.findall("./front/article-meta/title-group/article-title"):          # for elm in root.findall("./body/sec[@id='sec-1']/p"):
    st.write(elm.text)                                                              #     st.write(elm.text)
    st.write()                                                                      # not this as abstract is like this too

# tahun
for elm1 in root.findall("./front/article-meta/pub-date/year"):
    st.write(elm1.text)

# abstrak
for elm in root.findall("./front/article-meta/abstract/"):
    st.write(elm.text)

# if english wipe out, go:
# crawl that less than 10 WORDS !!, if yes:
#  then use other scenario (normal pdf text read)

from abstrak import pdfparser
import re
file_path = file_path.replace('cermxml', 'pdf')
parsed = pdfparser(file_path)
pattern = re.compile('(?<=abstrak)(.*)(?=kata kunci)', re.IGNORECASE)
abstract = re.findall(pattern, parsed)
