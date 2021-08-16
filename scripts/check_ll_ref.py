import json

with open('../data/nblist_ll.json') as json_file:
    data_ll = json.load(json_file)
    
with open('../data/nblist_ref.json') as json_file:
    data_ref = json.load(json_file)
    
for key in data_ref.keys():
    s1 = set(data_ref[key])
    s2 = set(data_ll[key])
    if not s1.issubset(s2):
        print("linked list did not captured all particles of interest at {}".format(key))