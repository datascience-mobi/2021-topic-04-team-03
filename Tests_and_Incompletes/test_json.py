import json
with open('values.json', 'r') as my_file:
    data = json.load(my_file)
    print(data["no preprocessing"]["NIH3T3"]["Dice Score"])

