from langdetect import detect, detect_langs
import re

def get_matching_pattern(list):            ## Function name changed
        for item in list:
            match = re.search('id', str(item), re.IGNORECASE)
            # st.write(match)   ## Use re.search method
            if match:                              ## Correct the indentation of if condition
                return match                       ## You also don't need an else statement
            match = re.search('de', str(item), re.IGNORECASE)
            # st.write(match)   ## Use re.search method
            if match: 
                return match 

def language_detection(text, 
                        method = "single"):

    """
    @desc: 
    - detects the language of a text
    @params:
    - text: the text which language needs to be detected
    - method: detection method: 
        single: if the detection is based on the first option (detect)
    @return:
    - the langue/list of languages
    """

    # if(method.lower() != "single"):
    #     result = detect_langs(text)

    # else:
    #     result = detect(text)
    result = detect_langs(text)

    return result

def mod_lang_detect_match(text):
    result = detect_langs(text)

    for item in result:
            match = re.search('id', str(item), re.IGNORECASE)
            # st.write(match)   ## Use re.search method
            if match:                              ## Correct the indentation of if condition
                return match                       ## You also don't need an else statement
            match = re.search('de', str(item), re.IGNORECASE)
            # st.write(match)   ## Use re.search method
            if match: 
                return match 
