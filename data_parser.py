import json
import re
from collections import Counter

# bag of words data and associated label
class Classification:
	def __init__(self, bag, label):
		self.bag = bag
		self.label = label

# list of Classification objects storing parsed product text and associated label
bag_list = []

# data outputted by this file
idf = Counter() # idf maps in how many labels does a given word appear
label_list = []
final_list =[]
common_list = []

# helpers
m = {}

def read_input():
	count  = 0
	# read each label in amazon_products and build 
	with open('./dataset/amazon_products') as f:
		for line in f:
			# running through first 300 labels in amazon data set
			if count <= 300:
				count += 1
				item = json.loads(line[:-2])
				size = len(item['Item']['PrunedEditorialReviews'])
				for i in range(size):
					words = item['Item']['PrunedEditorialReviews'][i]['Content'].lower()
					# words = re.sub(r'[^a-zA-Z0-9 ]', '', words).split()
					words = re.sub(r'[^a-zA-Z ]', '', words).split()
					bag = Counter(words)
					idf.update(bag.keys())
					# only one label for now
					label = item['Item']['ItemAttributes']['ProductGroup']
					bag_list.append(Classification(bag, label))
			else:
				break
	f.close()

def generate_structures():
	# common list
	# common_list = []
	common_set = set()
	for i in xrange(len(bag_list)):
		bag = bag_list[i].bag
		for key in bag.keys():
			common_set.add(key)
	global common_list
	common_list = list(common_set)

	# map tracks for each key where it occurs in common list
	# m = {}
	for i in range(len(common_list)):
		m[common_list[i]] = i

	# final list
	size = len(common_list)
	array = [0]*size
	# final_list =[]
	for i in range(len(bag_list)):
		final_list.append(list(array))
		cur_bag = bag_list[i].bag
		for key in cur_bag:
			rev_index = m[key]
			final_list[i][rev_index] = cur_bag[key]

	# label list
	for i, o in enumerate(bag_list):
		label_list.append(o.label)

	# map label list into index dictionary
	# done on other end


def write_files():
	# common list is mapping at what index what word occurs
	f = open('index_list.dat', 'w')
	f.write('\n'.join(common_list))
	f.close()

	# label list is mapping for that given text what was the output label
	f = open('label_list.dat', 'w')
	f.write('\n'.join(label_list))
	f.close()

	# feature vector maps for each label what were the counts for a given feature (word)
	f = open('feature_vector.dat', 'w')
	for i in range(len(final_list)):
		s = ' '.join(str(val) for val in final_list[i])+'\n'
		f.write(s)
	f.close()

	# split idf keys and values
	f = open('idf_key.dat', 'w')
	g = open('idf_val.dat', 'w')
	for key in idf.keys():
		f.write(str(key) + "\n")
		g.write(str(idf[key]) + "\n")
	f.close()
	g.close()


read_input()
generate_structures()
write_files()