import json

reverse_hierarchy_file_data = open("./data/task1_amazon_book_first_level_hierarchy.json").read()
reverse_hierarchy_json_data = json.loads(reverse_hierarchy_file_data)

x = list()
y = list()

count = 0

with open('./data/amazon_products') as f:
    for line in f:
        if count <= 3:
            item = json.loads(line[:-2])
            productGroup = item["Item"]["ItemAttributes"]["ProductGroup"]
            if productGroup == "Book":
                count += 1
                # Add each review as a datapoint
                for review in item["Item"]["PrunedEditorialReviews"]:
                    val = review["Content"].lower()
                    x.append(val)
                    
                    # Make list of labels for this review
                    labels = []
                    for node in item["Item"]["BrowseNodes"]["BrowseNode"]:
                        labels.append(reverse_hierarchy_json_data[node["Name"]])
                    y.append(labels)
        else:
            break
                

#for i in x:
#    print i
#    print "+++++++++++"
#    print "+++++++++++"
#print "======================"
#print "======================"
#print "======================"
#print "======================"
#print y
