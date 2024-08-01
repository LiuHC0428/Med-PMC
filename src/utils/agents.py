import os
import re
import pdb
import json
from openai import OpenAI
from utils.patient_conversation import get_patient_template, get_patient_prompt, get_actor_prompt, get_actor_plus_prompt
from utils.doctor_conversation import get_doctor_template, get_doctor_prompt
from utils.openai_utils import data_initialization, split_chinese_medicalinfo_and_question
from utils.general_utils import hisotry2str,history_detection


class Agent:
    def __init__(self, args, model, conv):
        self.args = args
        self.model = model
        self.conv =conv
    
    def postprocess(self, outputs):
        outputs = outputs.split(self.conv.roles[0], 1)[0]
        outputs = outputs.split(self.conv.roles[1], 1)[0]
        
        parts = re.split(r'[A-Z]{4,}:[^:]', outputs)
        if len(parts) > 1:
            outputs = parts[0]

        return outputs.rstrip("\n " + self.conv.roles[0] + self.conv.roles[1])

    def generate(self):
        pass

    def log(self):
        self.model.log()

class Doctor_Agent(Agent):
    def __init__(self, args, model, conv):
            super().__init__(args, model, conv)

    def postprocess(self, outputs):
        outputs = outputs.split(self.conv.roles[0], 1)[0]
        outputs = outputs.split(self.conv.roles[1], 1)[0]
        if "?" in outputs:
            outputs = outputs.split("?", 1)
            if len(outputs) > 1:
                outputs = outputs[0] + "?"
        elif "？" in outputs:
            outputs = outputs.split("？", 1)
            if len(outputs) > 1:
                outputs = outputs[0] + "？"
        
        parts = re.split(r'[A-Z]{4,}:[^:]', outputs)
        if len(parts) > 1:
            outputs = parts[0]

        return outputs
    
    def generate(self, data, cot_data, turn_id=0, bar=None):
        if len(data["history"]) < self.args.max_turn and (len(data["history"]) == 0 or data["history"][-1]["state"] != "D") and (len(data["history"]) == 0 or "patient" in data["history"][-1]):
            conv = self.conv.copy()
            conv.system_prompt_init(get_doctor_prompt(self.args.doctor_prompt_id))

            if self.args.cot == True:
                conv.init_history(cot_data["history"], first_key="doctor", second_key="patient")

            conv.init_history(data["history"], first_key="doctor", second_key="patient")

            conv.append_message(conv.roles[0], None)
            prompt = conv.get_prompt()
            if self.args.cot == True:
                prompt += "Let's think step by step. "
            images = []
            
            if len(data["history"]) >= 1:
                if data["history"][-1]['patient'].find('<\\image') != -1:
                    pattern = r'<\\image(.*?)>'
                    matches = re.findall(pattern, data["history"][-1]['patient'], re.DOTALL)
                    for match in matches:
                        image =  self.args.image_path + match
                        if os.path.exists(image):
                            images.append(image)

            outputs = ""
            while outputs == "":
                outputs = self.model.generate(prompt,images)
                if outputs.find(conv.roles[0]) <2:
                    outputs = outputs[:7].replace(conv.roles[0],'') + outputs[7:]

                outputs = self.postprocess(outputs)

                if outputs == "":
                    print("Retrying...")
                    pdb.set_trace()

            data["history"].append({"doctor": outputs})
        
            if self.args.debug:
                print(f"============== Doctor Turn {turn_id}===================")
                print(f"Sample: {data['id']}")
                print(f"Prompt: {prompt}")
                print(f"Output: {outputs}")
                pdb.set_trace()

        if bar is not None:
            bar.next()

class Patient_Agent(Agent):
    def __init__(self, args, model, conv):
        super().__init__(args, model, conv)
    
    def get_patient_info(self, whole_questions):

        return whole_questions
    
    def generate(self, data, turn_id=0, bar=None):
        # Four State Detection
        if data['history'][-1]['state'].find('B-A-') == -1:
            if len(data["history"]) <= self.args.max_turn and (data["history"][-1]["state"] != "D") and "patient" not in data["history"][-1].keys():
                        state_prompt = get_patient_prompt(self.args.patient_prompt_id, data["history"][-1]["state"])
                        if state_prompt: 
                            conv = self.conv.copy()
                            conv.system_prompt_init(state_prompt)
                            
                            conv.init_history(data["history"], turn=self.args.patient_history_len, first_key="doctor", second_key="standard_patient")
                            conv.append_message(conv.roles[1], None)
                            if data["history"][-1]["memory"] != '':
                                prompt = conv.get_prompt(patient_info=self.get_patient_info(data["history"][-1]["memory"]))
                            else:
                                if data['history'][-1]['state'] == '0':
                                    prompt = conv.get_prompt(patient_info=self.get_patient_info(data['Text']['Chief Complaint']))
                                else:
                                    prompt = conv.get_prompt(patient_info=self.get_patient_info(data['文本']))

                            outputs = self.model.generate(prompt)

                            outputs = self.postprocess(outputs)

                            pattern = r'<\\image(.*?)>'
                            matches2 = re.findall(pattern,outputs, re.DOTALL)
                            for match in matches2:
                                image =  self.args.image_path + match
                                if not os.path.exists(image):
                                    outputs = outputs.replace('<\\image'+match+'>','')
                                else:
                                    # pdb.set_trace()
                                    if self.args.only_text == True:   
                                        outputs = outputs.replace('<\\image'+match+'>',data['Image'][match])
                            
                            data["history"][-1]["standard_patient"] = outputs

                            
                            # actor
                            if self.args.actor_simulator is True:
                                if self.args.actor_plus == 'None':
                                    actor_prompt = get_actor_prompt(data['文本'])
                                else:
                                    actor_prompt = get_actor_plus_prompt(data['文本'],self.args.actor_plus)

                                conv = self.conv.copy()
                                conv.system_prompt_init(actor_prompt)

                                prompt = conv.get_prompt(patient_info=self.get_patient_info(data["history"][-1]["standard_patient"]))

                                outputs_actor = self.model.generate(prompt)

                                data["history"][-1]["patient"] = outputs_actor
                            else:
                                data["history"][-1]["patient"] = outputs

                            if self.args.debug:
                                print(f"============== Patient Turn {turn_id}===================")
                                print(f"Sample: {data['id']}")
                                print(f"State: {data['history'][-1]['state']}")
                                print(f"Memory: {data['history'][-1]['memory']}")
                                print(f"Prompt: {prompt}")
                                print(f"Output: {outputs}")
                                self.log()
                                pdb.set_trace()
                
                if bar is not None:
                    bar.next()
        else:
            if len(data["history"]) <= self.args.max_turn and (data["history"][-1]["state"] != "D") and "patient" not in data["history"][-1].keys():
                exam_memory = ''

                state_prompt = get_patient_prompt(self.args.patient_prompt_id, 1)
                if state_prompt: 
                    conv = self.conv.copy()
                    conv.system_prompt_init(state_prompt)
                    
                    conv.init_history(data["history"], turn=1, first_key="doctor", second_key="standard_patient")
                    conv.append_message('[Technologist]', None)
                    prompt = conv.get_prompt([])
                    outputs = self.model.generate(prompt)

                    outputs = self.postprocess(outputs)
                exam_memory = exam_memory + outputs + '\n'

                state_prompt = get_patient_prompt(self.args.patient_prompt_id, 2)
                if state_prompt: 
                    conv = self.conv.copy()
                    conv.system_prompt_init(state_prompt)
                    
                    conv.init_history(data["history"], turn=0, first_key="doctor", second_key="standard_patient")
                    conv.append_message('[Technologist]', None)
                    if data["history"][-1]["memory"] != '':
                        prompt = conv.get_prompt(patient_info=self.get_patient_info(data["history"][-1]["memory"]) + '<category of examination>：' + outputs)
                    else:
                        prompt = conv.get_prompt(patient_info=self.get_patient_info(data['检查报告']) + '<category of examination>：' + outputs)
                    outputs = self.model.generate(prompt)

                    outputs = self.postprocess(outputs)


                    pattern = r'<\\image(.*?)>'
                    matches2 = re.findall(pattern,outputs, re.DOTALL)
                    for match in matches2:
                        image =  self.args.image_path + match
                        if not os.path.exists(image):
                            outputs = outputs.replace('<\\image'+match+'>','')
                        else:
                            # pdb.set_trace()
                            if self.args.only_text == True:   
                                outputs = outputs.replace('<\\image'+match+'>',data['Image'][match])
                            if self.args.no_mm == True:
                                outputs = outputs.replace('<\\image'+match+'>','')

                    data["history"][-1]["patient"] = outputs

                    exam_memory = exam_memory + outputs + '\n'

                    data["history"][-1]["exam_memory"] = exam_memory
                    
                    
                    if self.args.debug:
                        print(f"============== Patient Turn {turn_id}===================")
                        print(f"Sample: {data['id']}")
                        print(f"State: {data['history'][-1]['state']}")
                        print(f"Memory: {data['history'][-1]['memory']}")
                        print(f"Prompt: {prompt}")
                        print(f"Output: {outputs}")
                        self.log()
                        pdb.set_trace()
            
            if bar is not None:
                bar.next()


class StateDetect_Agent(Agent):
    def __init__(self, args, model, conv=None, state_num=5):
        super().__init__(args, model, conv)

        self.state_num = state_num
        self.logit_bias = model.get_logit_bias(state_num)
        
        
        self.stageI_prompt =  """
During the consultation process, a doctor's questions can be categorized into five types:\n\
    (A) Inquiry: Doctors ask patients for information related to medical conditions, generally with words like 'please', 'please tell', '?', '?' or '?', and those not belonging to type (C) belong to this category.\n\
    (B) Examination: Doctors arrange patients for relevant examinations. Any suggestion by the doctor for the patient to undergo a certain medical examination belongs to this category.\n\
    (C) Other Topics: Questions from the doctor that do not pertain to the medical consultation scenario, and are unrelated to medical diseases, such as hobbies, movies, cuisine, etc.\n\
    (D) End: The doctor has completed the consultation and treatment recommendations have been given.\n\n\
Based on the descriptions of each question type above, identify the most appropriate category for the following doctor's question:\n\n\
Doctor's Question: {question}\n\
Question Type: ("""

        self.stageII_prompt = {
            "A": """
<Definition>:
    [Specific]: <Question> has a certain specific direction. When asking about symptoms, it should at least inquire about specific body parts, symptoms, sensations, or situations. When asking about examination results, it should mention specific body parts, specific examination items, or abnormal situations. Note that if it's about specific medical conditions, like medical history, family history, chronic illnesses, surgical history, etc., they are always considered [Specific]. Specifically, if the <Question> contain about demonstrative like "these" or "this", then it is related to the above and should belongs to the [Specific]
    [Broad]: <Question> such as "Where do you feel uncomfortable?" or "Where does it feel strange?" without any specific information direction are considered [Broad].

<Question>: {question}

Based on the <Definition>, determine whether the doctor's <Question> asks for [Specific] medical information from the patient or gives [Specific] advice. If so, directly output [Specific]. If not, output [Broad].
""",
            "B": """
<Definition>:
    [Specific]: <Advice> contains specific types of examinations or test (including but not limited to X-rays, MRI, biopsy, etc.), specific treatment plans (including but not limited to specific surgical treatments, exercises, diets, etc.), specific types of medication, etc.
    [Broad]: <Advice> broadly given without any specific examination/test, treatment plans, doctor's orders, exercises, diets and medication types is considered [Broad]. As long as any of the above information appears, <Advice> does not fall into this category.

<Advice>: {question}

Based on the <Definition>, determine whether the doctor's <Advice> asks for [Specific] medical information from the patient or gives [Specific] advice. If so, directly output [Specific]. If not, output [Broad].
"""}

        self.stageIII_prompt = {
            "A":"""
<Definition>:
    [Relevant Information]: <Patient Information> contains information asked in <Question>, including descriptions of having or not having the symptom, as long as there's relevant content.
    [No Relevant Information]: <Patient Information> does not contain information asked in <Question>, and there's no relevant content in the information.

<Patient Information>: {patient_info}

<Question>: {question}

Based on the <Definition>, determine whether <Patient Information> contains relevant information asked in <Question>. If [Relevant Information] is present, directly output the relevant text statement, ensuring not to include irrelevant content. If [No Relevant Information], then directly output [No Relevant Information].
""",
            "B":"""
<Definition>:
    [Relevant Information]: <Patient Information> contains results of the examinations or treatment plans suggested in <Advice>, including any results related to the suggested examination items and treatment plans.
    [No Relevant Information]: <Patient Information> does not contain results of the examinations or treatment plans suggested in <Advice>, including no mention of relevant examination items and treatment plans or no corresponding results.

<Patient Information>: {patient_info}

<Advice>: {question}

Based on the <Definition>, determine whether <Patient Information> contains relevant information about the measures suggested in <Advice>. If [Relevant Information] is present, directly output the relevant text statement, ensuring not to include irrelevant content. If [No Relevant Information], then directly output [No Relevant Information].
"""}

        else:
            raise NotImplementedError

    def generate_stageI(self, data, turn_id, bar=None):
        if "state" not in data["history"][-1].keys():
            question = data["history"][-1]["doctor"]
            prompt = self.stageI_prompt.format(question=question)

            if len(data["history"]) == 1:
                data["history"][-1]["state"] = '0'
            else:
                 while True:
                    outputs = self.model.generate(prompt)
                    if outputs.find('A') != -1:
                        data["history"][-1]["state"] = 'A'
                        break
                    elif outputs.find('B') != -1:
                        data["history"][-1]["state"] = 'B'
                        break
                    elif outputs.find('C') != -1:
                        data["history"][-1]["state"] = 'C'
                        break
                    elif outputs.find('D') != -1:
                        data["history"][-1]["state"] = 'D'
                        break
                    else:
                        print(outputs)
                        pdb.set_trace()
                        break

                
        
        if bar is not None:
            bar.next()
    
    def generate_stageII(self, data, turn_id, bar=None):
        if data["history"][-1]["state"] not in ["A", "B"]:
            return
        
        question = data["history"][-1]["doctor"]

        prompt = self.stageII_prompt[data["history"][-1]["state"]].format(question=question)
        outputs = self.model.generate(prompt)


        if outputs == "[Broad]" or outputs == "Broad":
            data["history"][-1]["state"] += "-B"
        elif outputs == "[Specific]" or outputs == "Specific":
            data["history"][-1]["state"] += "-A"
        else:
            # print("stageII output error!")
            # pdb.set_trace()
            data["history"][-1]["state"] += "-B"

        
        
        if bar is not None:
            bar.next()
    
    def generate_stageIII(self, data, turn_id, bar=None):
        if data["history"][-1]["state"] not in ["A-A", "B-A"]:
            if "memory" not in data["history"][-1].keys():
                data["history"][-1]["memory"] = ""
            return
        
        question = data["history"][-1]["doctor"]
        patient_info = {'A':data["文本"],'B':data['检查报告']}

        prompt = self.stageIII_prompt[data["history"][-1]["state"][0]].format(patient_info=patient_info[data["history"][-1]["state"][0]], question=question)
        outputs = self.model.generate(prompt)


        if outputs == "[No Relevant Information]" or outputs == "No Relevant Information":
            data["history"][-1]["state"] += "-B"
            data["history"][-1]["memory"] = ""
        else:
            data["history"][-1]["state"] += "-A"
            data["history"][-1]["memory"] = outputs

        


        
        if bar is not None:
            bar.next()
        
    def generate(self, data, turn_id, detect_type="stageI", bar=None):
        assert detect_type in ["stageI", "stageII", "stageIII"], f"detect_type: {detect_type} is not defined!"

        if detect_type == "stageI":
            self.generate_stageI(data, turn_id, bar)
        elif detect_type == "stageII":
            self.generate_stageII(data, turn_id, bar)
        elif detect_type == "stageIII":
            self.generate_stageIII(data, turn_id, bar)

        if self.args.debug:
            self.log()


class Dignosis_Agent(Agent):
    def __init__(self, args, model, conv=None, candidates_num=5):
        super().__init__(args, model, conv)
        self.state_num = candidates_num
    
        self.only_diagnosis = args.only_diagnosis
        if args.only_diagnosis is not True:
            self.prompt =  "Please generate a medical case report based on the following dialogue, including patient information, present illness, past medical history, examination items and results, diagnosis, and treatment plan.\n\
        **dialogue:**\n{conversations}\n\
        **medical case report:**("
        else:
            self.prompt =  "Please generate the results of the diagnosed condition directly without any analysis based on the following dialogue.\n\
        **dialogue:**\n{conversations}\n\
        **diagnosis**("

    def generate(self, data, bar=None):

        conversations = hisotry2str(data["history"])
        prompt = self.prompt.format(conversations=conversations)
        outputs = self.model.generate(prompt)
        if self.only_diagnosis is not True:
            data["diagnosis_self"] = outputs
        else:
            data["diagnosis_only"] = outputs

        if self.args.debug:
                    print(f"============== Diagnosis===================")
                    print(f"Sample: {data['id']}")
                    print(f"Prompt: {prompt}")
                    print(f"Output: {outputs}")
                    pdb.set_trace()

        if bar is not None:
            bar.next()