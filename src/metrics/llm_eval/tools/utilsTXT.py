import os
import shutil

def utilstxt(txt_path):
    consultation= []
    examination= []
    diagnosis= []
    treatment= []
    cskill= []
    logic = []
    with open(txt_path, 'r', encoding='utf-8') as efile:
        filestr = efile.read()

    filestr = filestr.lower()
    filestr +="\n"
    result = filestr.split("--------------------------")
    test = 0
    for res in result:
        count=0
        while res.find("score:")!=-1:
            position = res.find("score:")
            position+=7
            res = res[position:]
            vice_pos = 0 
            while (res[vice_pos] !="\n" and res[vice_pos] !="*"):
                vice_pos+=1
            try:  
                match count:
                    case 0:
                        consultation.append(float(res[:vice_pos]))
                    case 1:
                        examination.append(float(res[:vice_pos]))
                    case 2:
                        cskill.append(float(res[:vice_pos]))
                    case 3:
                        diagnosis.append(float(res[:vice_pos]))
                    case 4:
                        treatment.append(float(res[:vice_pos]))
                    case 5:
                        logic.append(float(res[:vice_pos]))
                    case _:
                        print("RuntimeError: Unknow case")
            except:
                # print(e)
                print("\nError in case: "+str(test)+"\n")
            res = res[vice_pos:]
            count+=1
        test+=1
    
    return consultation, examination, cskill, diagnosis, treatment, logic

def utilsmm(txt_path):
    total_behavior = []
    pic_analysis = []
    info_add = []

    with open(txt_path, 'r', encoding='utf-8') as efile:
        filestr = efile.read()
    
    filestr = filestr.lower()
    filestr +="\n"

    result = filestr.split("--------------------------")

    test = 0
    for res in result:
        count=0
        while res.find("score:")!=-1:
            position = res.find("score:")
            position+=7
            res = res[position:]
            vice_pos = 0 
            while (res[vice_pos] !="\n" and res[vice_pos] !="*" and res[vice_pos] !="'"):
                vice_pos+=1
            try:  
                match count:
                    case 0:
                        total_behavior.append(float(res[:vice_pos]))
                    case 1:
                        pic_analysis.append(float(res[:vice_pos]))
                    case 2:
                        info_add.append(float(res[:vice_pos]))
                    case _:
                        print("RuntimeError: Unknow case")
            except:
                # print(e)
                print("\nError in case: "+str(test)+"\n")
            res = res[vice_pos:]
            count+=1
        test+=1

    return total_behavior, pic_analysis, info_add
