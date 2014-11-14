import json
import re
from collections import Counter
import math

reverse_hierarchy_file_data = open("./data/task1_amazon_book_first_level_hierarchy.json").read()
reverse_hierarchy_json_data = json.loads(reverse_hierarchy_file_data)


x = [] # list of datapoints
y = [] # list of classifications

word_to_index = {} # dictionary mapping words to numbers
index_to_word = [] # list mapping index back to word

idf_counter = Counter() # idf maps in how many labels does a given word appear
review_count = 0 # number of documents

EXAMPLES_TO_PARSE = 2500
count = 0

def json_get(obj, key):
    # safe parsing of json object
    return obj.get(key, -1) if type(obj) == dict else -1

FILE_AMAZON_PRODUCTS = './data/amazon_products'
FILE_TO_OPEN = FILE_AMAZON_PRODUCTS

with open(FILE_TO_OPEN) as f:
    for line in f:
        if count <= EXAMPLES_TO_PARSE:
            item = json.loads(line[:-2]) # delimited by Ctrl+A
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
                            
                    review_count += 1
                    x.append(bag)
                    y.append(list(set(labels))) # trim duplicate labels
                    idf_counter.update(bag.keys())
                # end for
            # end if
        # end if
        else:
            break
    # end for
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

f = open('./stats/task2_amazon_stats.dat', 'w')
f.write('review_count = ' + str(review_count))
f.close()
