import json


def transcase(case_data):
    case_format=""
    for key in case_data:
        if key=="history":
            case_format+="\nDialogues:\n"
            for history_key in case_data[key]:
                for k in history_key:
                    if k=="doctor" or k=="patient":
                        case_format+=k+":"+str(history_key[k])+"\n"
            case_format+="--------------------------\n"
        elif key=="diagnosis_self":
            case_format+="\nRecords summarized by doctor:"+str(case_data[key])+"\n"
        elif key=="Text":
            case_format+="\nCase info:\n"
            for text_key in case_data[key]:
                case_format+=str(text_key)+":"+str(case_data[key][text_key])+"\n"
            case_format+="--------------------------\n"
        else:
            continue
    return case_format

def get_conversation(case_data):
    conversation = ""
    for key in case_data:
        if key == "history":
            conversation += "\nDialogues:\n"
            for history_key in case_data[key]:
                for k in history_key:
                    if k == "doctor" or k == "patient":
                        conversation += k + ":" + str(history_key[k]) + "\n"
    return conversation

def get_case_history(case_data):
    case_history = ""
    for key in case_data:
        if key == "history":
            case_history += "\nDialogues:\n"
            for history_key in case_data[key]:
                for k in history_key:
                    if k == "doctor" or k == "patient":
                        case_history += k + ":" + str(history_key[k]) + "\n"
        elif key == "diagnosis_self":
            case_history += "\nRecords summarized by doctor:" + str(case_data[key]) + "\n"
        elif key == "文本":
            case_history += "\nCase info:" + str(case_data[key]) + "\n"
        else:
            continue
    return case_history

def get_exam(case_data):
    case_exam = ""
    for key in case_data:
        if key == "history":
            case_exam += "\nDialogues:\n"
            for history_key in case_data[key]:
                for k in history_key:
                    if k == "doctor" or k == "patient":
                        case_exam += k + ":" + str(history_key[k]) + "\n"
        elif key == "diagnosis_self":
            case_exam += "\nRecords summarized by doctor:" + str(case_data[key]) + "\n"
        elif key == "检查报告":
            case_exam += "\nExamination reports:" + str(case_data[key]) + "\n"
        elif key == "图片":
            case_exam += "\nX-ray photo:\n"
            for img in case_data[key]:
                case_exam += str(img) + ":" + str(case_data[key][img]) + "\n"
        else:
            continue
    return case_exam

def get_mm(case_data_mm, case_data_text):
    case_mm = ""
    for key in case_data_mm:
        if key == "history":
            case_mm += "\nDialogues:\n"
            for history_key in case_data_mm[key]:
                for k in history_key:
                    if k == "doctor" or k == "patient":
                        case_mm += k + ":" + str(history_key[k]) + "\n"                    
        else:
            continue

    case_text = ""
    for key in case_data_text:
        if key == "history":
            case_text += "\nDialogues:\n"
            for history_key in case_data_text[key]:
                for k in history_key:
                    if k == "doctor" or k == "patient":
                        case_text += k + ":" + str(history_key[k]) + "\n"                    
        else:
            continue

    
    pic_position = case_mm.find("<\\image")
    if(pic_position == -1):
        return "invalid", "invaild"
    begin_pos = case_mm.rfind("patient:",0, pic_position)
    end_pos = case_mm.find("patient:",pic_position)
    case_mm = case_mm[begin_pos:end_pos]

    text_beg_pos = begin_pos
    text_end_pos = case_text.find("patient:",text_beg_pos+10)

    case_text = case_text[text_beg_pos:text_end_pos]

    return case_mm, case_text



if __name__ == '__main__':
    case_all = json.load(open('./data/M3_30_Result_standard/gpt4o_qwen-max.json', 'r', encoding='utf-8'))
    # case_text_only = json.load(open('./data/M3_30_Result_standard_text/gpt4o_qwen-max.json', 'r', encoding='utf-8'))
    case_format=""
    case_text=""
    count = 1
    for case_data in case_all:
        case_format += f'\nCase {count}:\n'
        case_temple =""
        case_format += transcase(case_data)
        count += 1
    with open('./sundry/mmtest.txt', 'w', encoding='utf-8') as efile:
        efile.write(case_format)
