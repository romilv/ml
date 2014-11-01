import json

hierarchy = {}

hierarchy_file_data = open("AmazonHeirarchy.json").read()
hierarchy_json_data = json.loads(hierarchy_file_data)

def recursivelyAddLabel(root):
    label_set = set()
    label_set.add(root.get("Name"))
    children = root.get("Children", [])
    for child in children:
        label_set = label_set.union(recursivelyAddLabel(child))
    return label_set

for item in hierarchy_json_data:
    name = item.get("Name")
    if name == "Books":
        children = item.get("Children")
        for child in children:
            subname = child.get("Name")
            hierarchy[subname] = recursivelyAddLabel(child)

reverse_hierarchy = {}
label_map = {}
n = 0

for key, label_set in hierarchy.iteritems():
    for label in label_set:
        reverse_hierarchy[label] = n
    label_map[key] = n
    n += 1

with open('AmazonBookFirstLevelHierarchy.json', 'w') as outfile:
    json.dump(reverse_hierarchy, outfile)
outfile.close()

with open('AmazonBookLabelMap.dat', 'w') as outfile:
    json.dump(label_map, outfile)
outfile.close()