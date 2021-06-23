import json
list = []
with open('values.json', 'r') as my_file:
    data = json.load(my_file)
    for methode in data:
        list.append(methode)

