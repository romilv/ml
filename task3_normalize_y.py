import json
from collections import Counter

def normalize_y(file_name):
	label_counter = Counter()

	y_file_data = open(file_name).read()
	y = json.loads(y_file_data)
	normalized_y = list()
	# print y
	# print len(y)
	for i, arr in enumerate(y):
		min_j = -1
		min_val = float('inf')
		for j, elem in enumerate(arr):
			if label_counter[elem] < min_val:
				min_val, min_j = label_counter[elem], j
		# print arr[min_j]
		label_counter[arr[min_j]] += 1
		# label_counter[arr[min_j]]
		# print i, arr
		normalized_y.append(arr[min_j])

	# print normalized_y
	print label_counter
	return normalized_y
	# print len(normalized_y)

# def 

# normalize_y("./data/task2_y.json")
# y_file_data = open("./data/task2_y.json").read()