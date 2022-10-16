import io

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

import re
import string

def pdfparser(pdffile):
    with open(pdffile, mode='rb') as f:
    #fp = open(data, 'rb')
        rsrcmgr = PDFResourceManager()
        retstr = io.StringIO()
        codec = 'utf-8'
        laparams = LAParams()
        data =[]
        device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
        # Create a PDF interpreter object.
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        # Process each page contained in the document.
        for page in PDFPage.get_pages(f):
            interpreter.process_page(page)
            data = retstr.getvalue()
            #print(data)

        # Cleaning the data
        data = data.lower()
        data = re.sub('\[*?\]', '', data)
        data = re.sub('[%s]' % re.escape(string.punctuation), '', data)
        data = re.sub('\w*\d\w*', '', data)
        data = data.replace("\n", "")
        # print(data)

        return data

    # paragraph = "Abstract"
def abstractExtraction(text,paragraph):

    count = 0
    para=""
    text=text.replace('\n\n+', '\n')
    text=text.replace('\s\s\s+', '\n')
    for i in re.split(r'\n+', text):
        p = re.compile('(?<!\S)abstrak', re.IGNORECASE)
        p1 = re.compile('abstrak')
        if(str(p1.match(i)))=='None':
            if str(p.match(i))!='None':
                count=1
            elif count == 1:
                if str(re.compile('\d' + '.*' + '\s*' + '(pendahuluan|abstract|kata kunci)', re.IGNORECASE).match(i))!='None':
                    return para
                elif str(re.compile('X|IV|V?I{0,3}' + '.*' + '\s*' + '(pendahuluan|abstract|kata kunci)', re.IGNORECASE).match(i))!='None':
                    return para
                else:
                    para =para+i
                    continue
    if(len(para)>1000):
        return 'None'
    else:
        return para

def between(value, a, b):
    # Find and validate before-part.
    pos_a = value.find(a)
    if pos_a == -1: return ""
    # Find and validate after part.
    pos_b = value.rfind(b)
    if pos_b == -1: return ""
    # Return middle part.
    adjusted_pos_a = pos_a + len(a)
    if adjusted_pos_a >= pos_b: return ""
    return value[adjusted_pos_a:pos_b]