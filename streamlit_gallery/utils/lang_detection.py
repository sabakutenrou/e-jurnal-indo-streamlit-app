from langdetect import detect, detect_langs

def language_detection(text, method = "single"):

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

    if(method.lower() != "single"):
        result = detect_langs(text)

    else:
        result = detect(text)

    return result