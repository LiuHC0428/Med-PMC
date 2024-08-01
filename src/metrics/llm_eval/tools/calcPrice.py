import json

def get_price():
    log_path = "./logs/openai.json"

    log = json.load(open(log_path, 'r'))

    total_price = 0

    for i in log:
        total_price += i["dollar_cost"]

    print(total_price)