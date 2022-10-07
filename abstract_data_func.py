import io
from io import StringIO

from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser

import re
import string

def convert_pdf_to_string(file_path):

    output_string = StringIO()
    with open(file_path, 'rb') as in_file:
        parser = PDFParser(in_file)
        doc = PDFDocument(parser)
        rsrcmgr = PDFResourceManager()
        device = TextConverter(rsrcmgr, output_string, laparams=LAParams())
        interpreter = PDFPageInterpreter(rsrcmgr, device)
        for page in PDFPage.create_pages(doc):
            interpreter.process_page(page)

    return(output_string.getvalue())

                
def convert_title_to_filename(title):
    filename = title.lower()
    filename = filename.replace(' ', '_')
    return filename


def split_to_title_and_pagenum(table_of_contents_entry):
    title_and_pagenum = table_of_contents_entry.strip()
    
    title = None
    pagenum = None
    
    if len(title_and_pagenum) > 0:
        if title_and_pagenum[-1].isdigit():
            i = -2
            while title_and_pagenum[i].isdigit():
                i -= 1

            title = title_and_pagenum[:i].strip()
            pagenum = int(title_and_pagenum[i:].strip())
        
    return title, pagenum




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
            # print(count)
    if(len(para)>1000):
        return 'None'
    else:
        return para