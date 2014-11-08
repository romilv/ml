import json
import re
from collections import Counter
import math

reverse_hierarchy_file_data = open("./data/task1_amazon_book_first_level_hierarchy.json").read()
reverse_hierarchy_json_data = json.loads(reverse_hierarchy_file_data)

word_to_index_file_data = open("./data/task2_word_to_index.json").read()
word_to_index = json.loads(word_to_index_file_data)


x = [] # list of datapoints
y = [] # list of classifications

idf_counter = Counter() # idf maps in how many labels does a given word appear

EXAMPLES_TO_PARSE = 1000
review_count = 0 # a product can have multiple reviews
count = 0

def json_get(obj, key):
    return obj.get(key, -1) if type(obj) == dict else -1

FILE_TO_OPEN = './data/Social_Conversations_AmazonLabel.json'

with open(FILE_TO_OPEN) as f:
    for line in f:
        if count <= EXAMPLES_TO_PARSE:
            item = json.loads(line)
            browseNodeList = json_get(json_get(item, 'Amazon_Browsenodes'), "BrowseNode")
            if browseNodeList == -1:
                continue
            label = list()
            for browseNode in browseNodeList:
                nameNode = json_get(browseNode, "Name")
                if nameNode != -1 and nameNode in reverse_hierarchy_json_data:
                    label.append(reverse_hierarchy_json_data[nameNode])
            
            if not label:
                continue
            count += 1

            review = item["conversation_text"] # social data has a single review per item
            
            words = review.lower()
            words = re.sub(r'[^a-zA-Z ]', '', words).split()
            amazon_mapped_words = []
            
            # convert words to indices
            for i, word in enumerate(words):
                word_index = word_to_index.get(word, -1)
                if word_index == -1:
                    continue
                amazon_mapped_words.append(word_index)
            # end for
            
            bag = Counter(amazon_mapped_words)
                    
            review_count += 1
            x.append(bag)
            y.append(list(set(label))) # trim duplicate labels
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
with open('./data/task3_social_tf_dict.json', 'w') as outfile:
    json.dump(x, outfile)
outfile.close()

# dump word_to_index to file
with open('./data/task3_social_y.json', 'w') as outfile:
    json.dump(y, outfile)
outfile.close()

# dump idf_counter to file
with open('./data/task3_social_tfidf2d_list.json', 'w') as outfile:
    json.dump(tfidf_list, outfile)
outfile.close()

f = open('./data/task3_social_stats.dat', 'w')
f.write('review_count = ' + str(review_count))
f.close()
