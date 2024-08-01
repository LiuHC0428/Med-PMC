import os
import json
import re

folder_path= '/M3_30_Result_actor'
save_folder = '/M3_30_Result_actor_nomm'
if os.path.exists(save_folder) is not True:
    os.mkdir(save_folder)
for path, dir_lst, file_lst in os.walk(folder_path):
    for file_name in file_lst:
        if file_name.find('huatuo-vision-7b') != -1:
            file_path = os.path.join(folder_path,file_name)
            save_path = os.path.join(save_folder,file_name)

            with open(file_path, "r", encoding="utf-8") as f:
                datas = json.load(f)

            new_datas = []
            for data in datas:
                new_data = data.copy()
                del(new_data['history'])
                del(new_data['diagnosis_self'])

                out_exam = ''
                exams = new_data['Text']['Examinations'].split('\n\n')
                for exam in exams:
                    if exam.find('<\\image') == -1:
                        out_exam += exam + '\n\n'
                if out_exam == '':
                    out_exam = 'None'
                        
                new_data['Text']['Examinations'] = out_exam
                new_data['检查报告']  = out_exam

                new_data['history'] = []
                for dialog in data['history']:

                    pattern = r'<\\image(.*?)>'
                    if dialog['state'] =='D':
                        new_data['history'].append(dialog)
                        break
                    else:
                        matches = re.findall(pattern, dialog['patient'] , re.DOTALL)
                        if matches == []:
                            new_data['history'].append(dialog)
                        else:
                            break

                new_datas.append(new_data)
            
            with open(save_path,"w", encoding="utf-8") as f:
                json.dump(new_datas,f, indent=4, ensure_ascii=False)


                
