import json
import re
from collections import Counter
import math

reverse_hierarchy_file_data = open("./data/task1_amazon_book_first_level_hierarchy.json").read()
reverse_hierarchy_json_data = json.loads(reverse_hierarchy_file_data)

# list of datapoints
x = []
# list of classifications
y = []


# dictionary mapping words to numbers
word_to_index = {}
# list mapping index back to word
index_to_word = []

idf_counter = Counter() # idf maps in how many labels does a given word appear

# number of documents
review_count = 0

count = 0

def json_get(obj, key):
    return obj.get(key, -1) if type(obj) == dict else -1

FILE_AMAZON_PRODUCTS = './data/amazon_products'
# FILE_SOCIAL_CONVERSATIONS = './data/Social_Conversations_AmazonLabel.json'
FILE_TO_OPEN = FILE_AMAZON_PRODUCTS

with open(FILE_TO_OPEN) as f:
    for line in f:
        if count <= 500:
            # if FILE_TO_OPEN == FILE_AMAZON_PRODUCTS:
            item = json.loads(line[:-2])
            # else:
                # item = json.loads(line)
                # print item
            productGroup = item["Item"]["ItemAttributes"]["ProductGroup"]
            if productGroup == "Book":
                count += 1
                # Add each review as a datapoint
                for review in item["Item"]["PrunedEditorialReviews"]:
                    words = review["Content"].lower()
                    words = re.sub(r'[^a-zA-Z ]', '', words).split()
                    
                    # convert words to indices
                    for i, word in enumerate(words):
                        word_index = word_to_index.get(word, len(word_to_index)+1)
                        word_to_index[word] = word_index
                        words[i] = word_index
                        if len(index_to_word) < word_index:
                            index_to_word.append(word)
                    # end for
                    
                    bag = Counter(words)
                    
                    # Make list of labels for this review
                    labels = []
                    nodes = json_get(json_get(json_get(item, "Item"), "BrowseNodes"), "BrowseNode")
                    # nodesB = json_get(json_get(json_get(item, "Item"), "Amazon_Browsenodes"), "BrowseNode")
                    # Amazon_Browsenodes
                    if nodes == -1:
                        # if nodesB == -1:
                        continue
                        # nodes = nodesB

                    for node in nodes:
                        if type(node) != dict:
                            continue
                        label = reverse_hierarchy_json_data.get(node["Name"], -1)
                        if label != -1:
                            labels.append(label)
                    
                    if not labels:
                        continue
                            
                    review_count += 1
                    x.append(bag)
                    y.append(list(set(labels))) # trim duplicate labels
                    # y.append(labels)
                    idf_counter.update(bag.keys())
        else:
            break
f.close()

# have idf_list[0] = 0 as a dummy since the first word is index 1
tfidf_list = [[0] for _ in range(review_count)]
for key, val in idf_counter.iteritems():
    idf = math.log( review_count / val )
    for i, review in enumerate(x):
        tfidf_list[i].append(review[key]*idf)

# dump word_to_index to file
with open('./data/task2_word_to_index.json', 'w') as outfile:
    json.dump(word_to_index, outfile)
outfile.close()

# dump index_to_word to file
with open('./data/task2_index_to_word.json', 'w') as outfile:
    json.dump(index_to_word, outfile)
outfile.close()

# dump word_to_index to file
with open('./data/task2_tf_dict.json', 'w') as outfile:
    json.dump(x, outfile)
outfile.close()

# dump word_to_index to file
with open('./data/task2_y.json', 'w') as outfile:
    json.dump(y, outfile)
outfile.close()

# dump idf_counter to file
with open('./data/task2_tfidf2d_list.json', 'w') as outfile:
    json.dump(tfidf_list, outfile)
outfile.close()

f = open('./data/task2_stats.dat', 'w')
f.write('review_count = ' + str(review_count))
f.close()
