import json
import re

EXAMPLES_TO_PARSE = 200
reverse_hierarchy_file_data = open("./data/task1_amazon_book_first_level_hierarchy.json").read()
reverse_hierarchy_json_data = json.loads(reverse_hierarchy_file_data)

def json_get(obj, key):
    return obj.get(key, -1) if type(obj) == dict else -1

def parse_amazon_book_objects():
	file_to_open = './data/amazon_products'
	file_to_write = './data/amazon_products_books'
	count = 0
	with open(file_to_open) as f:
		# with open(file_to_write) as g:
		with open(file_to_write, 'w') as outfile:
		    for line in f:
		        if count <= EXAMPLES_TO_PARSE:
		            item = json.loads(line[:-2]) # delimited by Ctrl+A
		            productGroup = item["Item"]["ItemAttributes"]["ProductGroup"]
		            if productGroup == "Book":
		                count += 1
		                outfile.write(line[:-2])
		                # outfile.write("\n\n\n")
		    # end for
		outfile.close()
	f.close()
	# end parse_amazon_book_objects

def parse_twitter_book_objects():
	file_to_open = './data/Social_Conversations_AmazonLabel.json'
	file_to_write = './data/Social_Conversations_AmazonLabel_Books.json'
	count = 0
	with open(file_to_open) as f:
		# with open(file_to_write) as g:
		with open(file_to_write, 'w') as outfile:
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
		            outfile.write(line)
		outfile.close()
	f.close()
	# end parse_amazon_book_objects
parse_amazon_book_objects()
parse_twitter_book_objects()