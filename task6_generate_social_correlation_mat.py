# write 31x31 correlation matrix for book labels for over 50K dataset

import json
import itertools
import math
import numpy

print "UNCOMMENT WRITE TO FILE IN LAST LINE"

def json_get(obj, key):
    # safe parsing of json object
    return obj.get(key, -1) if type(obj) == dict else -1

reverse_hierarchy_file_data = open("./data/task1_amazon_book_first_level_hierarchy.json").read()
reverse_hierarchy_json_data = json.loads(reverse_hierarchy_file_data)

FILE_TO_OPEN = './data/Social_Conversations_AmazonLabel.json'
EXAMPLES_TO_PARSE = 100000

numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json')))


arr = [0.0] * numLabels
corr_mat = numpy.array([list(arr) for _ in range(numLabels)])
count = 0
l_sum = 0
with open(FILE_TO_OPEN) as f:
    for line in f:
        if count <= EXAMPLES_TO_PARSE:
            item = json.loads(line)
            browseNodeList = json_get(json_get(item, 'Amazon_Browsenodes'), "BrowseNode")
            if browseNodeList == -1:
                continue
            labels = list()
            for browseNode in browseNodeList:
                nameNode = json_get(browseNode, "Name")
                if nameNode != -1 and nameNode in reverse_hierarchy_json_data:
                    labels.append(reverse_hierarchy_json_data[nameNode])

            if not labels:
                continue
            count += 1
            labels = list(set(labels))
            l_sum += len(labels)
            pairs = itertools.combinations(labels,2)
            for pair in pairs:
            	if pair[0] != pair[1]:
            		corr_mat[pair[0]][pair[1]] += 1.0
            		corr_mat[pair[1]][pair[0]] += 1.0
print count
print "l_sum", l_sum, (1.0*l_sum)/count
f.close()

# smooth and normalize
for idx,label_list in enumerate(corr_mat):
    smoothing_factor = max(label_list)*0.01 # 1 or (1/100) of max in row
    if smoothing_factor == 0:
        smoothing_factor = 1
    # row_count = sum(label_list)
    label_list += smoothing_factor
    total = sum(label_list)
    if total==0:
        continue
    # log_val = math.log(2.0 + row_count, 2)
    # print log_val
    corr_mat[idx]*=1.0/total
    # corr_mat[idx]*=1.0*log_val/total

# UNCOMMENT
numpy.save('./stats/social_correlation_mat', corr_mat)