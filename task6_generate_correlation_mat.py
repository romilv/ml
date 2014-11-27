# write 31x31 correlation matrix for book labels for over 50K dataset

import json
import itertools
import numpy

print "UNCOMMENT WRITE TO FILE IN LAST LINE"

def json_get(obj, key):
    # safe parsing of json object
    return obj.get(key, -1) if type(obj) == dict else -1

reverse_hierarchy_file_data = open("./data/task1_amazon_book_first_level_hierarchy.json").read()
reverse_hierarchy_json_data = json.loads(reverse_hierarchy_file_data)

FILE_TO_OPEN = './data/amazon_products'
EXAMPLES_TO_PARSE = 100000

numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json')))

arr = [0.0] * numLabels
corr_mat = numpy.array([list(arr) for _ in range(numLabels)])
count = 0
with open(FILE_TO_OPEN) as f:
	for line in f:
		if count <= EXAMPLES_TO_PARSE:
			item = json.loads(line[:-2]) # delimited by Ctrl+A
			productGroup = item["Item"]["ItemAttributes"]["ProductGroup"]
			if productGroup == "Book":
				count += 1
				for review in item["Item"]["PrunedEditorialReviews"]:
					labels = []
					nodes = json_get(json_get(json_get(item, "Item"), "BrowseNodes"), "BrowseNode")
					
					if nodes == -1:
						continue
					for node in nodes:
					    if type(node) != dict:
					        continue
					    label = reverse_hierarchy_json_data.get(node["Name"], -1)
					    if label != -1:
					        labels.append(label)

					if not labels:
					    continue

					pairs = itertools.combinations(labels,2)
					for pair in pairs:
						if pair[0] != pair[1]:
							corr_mat[pair[0]][pair[1]] += 1.0
							corr_mat[pair[1]][pair[0]] += 1.0

f.close()

# smooth and normalize
for idx,label_list in enumerate(corr_mat):
    smoothing_factor = max(label_list)*0.01 # 1 or (1/100) of max in row
    label_list += smoothing_factor
    total = sum(label_list)
    if total==0:
        continue
    corr_mat[idx]*=1.0/total

# UNCOMMENT

numpy.save('./stats/correlation_mat', corr_mat)