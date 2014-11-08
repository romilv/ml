import json
from collections import Counter

def normalize_y(file_name):
	# try to evenly distribute training data set assignment to each label in y 
	# eg file_name = "./data/task2_y.json"
	label_counter = Counter()
	normalized_y = list()
	
	y_file_data = open(file_name).read()
	y = json.loads(y_file_data)

	for i, arr in enumerate(y):
		# among all y candidates, assign to candidate that has been assigned least 
		# number of times previously
		min_j = -1
		min_val = float('inf')
		for j, elem in enumerate(arr):
			if label_counter[elem] < min_val:
				min_val, min_j = label_counter[elem], j
		label_counter[arr[min_j]] += 1
		normalized_y.append(arr[min_j])
	return normalized_y