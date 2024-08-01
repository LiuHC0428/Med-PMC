from jsonpath import jsonpath
import json
from tools.transform_caseinfo import *

def get_case_history(case_data):
    case_history = ""
    for key in case_data:
        if key == "history":
            case_history += "\nDialogues:\n"
            for history_key in case_data[key]:
                for k in history_key:
                    if k == "doctor" or k == "patient":
                        case_history += k + ":" + str(history_key[k]) + "\n"
        else:
            continue
    return case_history


def findNoPicSamples(case_data):
    targetString = get_case_history(case_data)
    return targetString.find("<\\image")

