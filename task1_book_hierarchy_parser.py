import json

HIERARCHY_FILE = "./data/AmazonHierarchy.json" # file used for all classifications
FILE_PATH = './data/'

def recursively_add_label(root): 
    # 
    label_set = set()
    label_set.add(root.get("Name"))
    children = root.get("Children", [])
    for child in children:
        label_set = label_set.union(recursively_add_label(child))
    return label_set
    # end recursively_add_label

def write_first_level_hierarchy(file_prefix, reverse_hierarchy):
    file_name =  FILE_PATH + file_prefix + '_first_level_hierarchy.json'
    with open(file_name, 'w') as outfile:
        json.dump(reverse_hierarchy, outfile)
    outfile.close()
    # end write_first_level_hierarchy

def write_label_map(file_prefix, label_map):
    file_name =  FILE_PATH + file_prefix + '_label_map.json'
    with open(file_name, 'w') as outfile:
        json.dump(label_map, outfile)
    outfile.close()
    # end write_label_map

def create_hierarchy_files(file_prefix):
    # file_prefix = task1_amazon_book
    hierarchy = {}
    hierarchy_file_data = open(HIERARCHY_FILE).read()
    hierarchy_json_data = json.loads(hierarchy_file_data)

    for item in hierarchy_json_data:
        name = item.get("Name")
        if name == "Books":
            children = item.get("Children")
            for child in children:
                subname = child.get("Name")
                hierarchy[subname] = recursively_add_label(child)

    reverse_hierarchy = {}
    label_map = {}
    n = 0

    for key, label_set in hierarchy.iteritems():
        for label in label_set:
            reverse_hierarchy[label] = n
        label_map[key] = n
        n += 1

    # write files
    write_first_level_hierarchy(file_prefix, reverse_hierarchy)
    write_label_map(file_prefix, label_map)

    # end create_hierarchy_files

# usage
if __name__ == "__main__":
    print "book_hierarchy_parser called directly"
    create_hierarchy_files('task1_amazon_book') # this will create first level and label map using HIERARCHY_FILE