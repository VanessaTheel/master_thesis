import pycld2 as cld2  # compact language detector library
# Helper exception for failed detection
class UnknownLanguageException(Exception):
    pass
# extracts confidence and language code from a choice tuple, which is returned by cld2
class Language(object):
    def __init__(self, choice):
        basic_name, code, confidence, bytesize = choice
        self.code = code
        self.confidence = float(confidence)
# detects the language of the text snippet
def detect_language(text):
    t = text.encode("utf-8")
    reliable, index, top_3_choices = cld2.detect(t, bestEffort=False)
    if not reliable:
        # retry with lower confidence (for short texts)
        reliable, index, top_3_choices = cld2.detect(t, bestEffort=True)
        if not reliable:
            raise UnknownLanguageException("We need more text - Unknown Language")
        else:
            pass
            # import sys
            # print("Detector is not able to detect the language reliably.", file=sys.stderr)
    # convert and return cld2 output as Language
    return [Language(x) for x in top_3_choices]