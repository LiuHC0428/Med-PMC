import json
import tools.utilsTXT as utxt

def m3(txt_path, json_path):
    cost_log = {
        "id": 0,
        "analysis": 0,
        "behavior": 0,
        "info_gain": 0
    }
    count = 0
    arr1, arr2, arr3=utxt.utilsmm(txt_path)
    for param1, param2, param3 in zip(arr1, arr2, arr3):
        count += 1
        with open(json_path, "r")as files:
            json_data = json.load(files)
            cost_log["id"] = count
            cost_log["analysis"] = param1
            cost_log["behavior"] = param2
            cost_log["info_gain"] = param3
            json_data.append(cost_log)
        with open(json_path, "w") as f:
                try:
                    json.dump(json_data, f)
                except:
                    print("Error in log")

def whole_huatuo(txt_path, json_path):
    cost_log = {
        "id": 0,
        "inquary": 0,
        "exam": 0,
        "diagnosis": 0,
        "treatment": 0,
        "cskill": 0,
        "logic": 0
    }
    count = 0
    arr1, arr2, arr3, arr4, arr5, arr6=utxt.utilstxt(txt_path)
    for param1, param2, param3, param4, param5, param6 in zip(arr1, arr2, arr3, arr4, arr5, arr6):
        count += 1
        with open(json_path, "r")as files:
            json_data = json.load(files)
            cost_log["id"] = count
            cost_log["inquary"] = param1
            cost_log["exam"] = param2
            cost_log["diagnosis"] = param3
            cost_log["treatment"] = param4
            cost_log["cskill"] = param5
            cost_log["logic"] = param6
            json_data.append(cost_log)
        with open(json_path, "w") as f:
                try:
                    json.dump(json_data, f)
                except:
                    print("Error in log")

def whole_normal(CEC, DT, logic, json_path):
    cost_log = {
        "id": 0,
        "inquary": 0,
        "exam": 0,
        "diagnosis": 0,
        "treatment": 0,
        "cskill": 0,
        "logic": 0
    }
    count = 0

    arr1, arr2, _, arr3, _, _ = utxt.utilstxt(CEC)
    arr4, arr5, _ = utxt.utilsmm(DT)
    arr6, _, _ =utxt.utilsmm(logic)

    for param1, param2, param3, param4, param5, param6 in zip(arr1, arr2, arr3, arr4, arr5, arr6):
        count += 1
        with open(json_path, "r")as files:
            json_data = json.load(files)
            cost_log["id"] = count
            cost_log["inquary"] = param1
            cost_log["exam"] = param2
            cost_log["diagnosis"] = param3
            cost_log["treatment"] = param4
            cost_log["cskill"] = param5
            cost_log["logic"] = param6
            json_data.append(cost_log)
        with open(json_path, "w") as f:
                try:
                    json.dump(json_data, f)
                except:
                    print("Error in log")


