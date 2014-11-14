import json
import mlpy
import math
import task3_normalize_y

x_file_name = './data/task2_tfidf2d_list.json'
x_file_data = open(x_file_name).read()
x = json.loads(x_file_data)

y_file_name = './data/task2_y.json'
y_trimmed = task3_normalize_y.normalize_y(y_file_name)

# we use 70% of x for training and 30% for testing
TRAIN_PERCENT = 0.7
split_index = int(math.floor(TRAIN_PERCENT * len(x)))

model = mlpy.LibLinear(solver_type='l2r_lr')
model.learn(x[:split_index], y_trimmed[:split_index])

train_num_correct = 0
test_num_correct = 0
num_labels = max(y_trimmed)

train_confusion = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
test_confusion = [[0 for _ in range(num_labels)] for _ in range(num_labels)]

# train error
for i in range(split_index):
    prediction = model.pred(x[i])
    actual = y_trimmed[i]
    if prediction == actual:
        train_num_correct += 1
    train_confusion[actual - 1][prediction - 1] += 1

# test error
for i in range(split_index, len(x)):
    prediction = model.pred(x[i])
    actual = y_trimmed[i]
    if prediction == actual:
        test_num_correct += 1
    test_confusion[actual - 1][prediction - 1] += 1

def print_matrix(matrix):
    for row in matrix:
        print row

# print "=========================================="
# print_matrix(train_confusion)
# print "=========================================="
# print_matrix(test_confusion)
# print "=========================================="

print train_num_correct, "/", split_index
print test_num_correct, "/", len(x) - split_index

with open('./data/task2_train_confusion.json', 'w') as outfile:
    json.dump(train_confusion, outfile)
outfile.close()

with open('./data/task2_test_confusion.json', 'w') as outfile:
    json.dump(test_confusion, outfile)
outfile.close()

f = open('./stats/task2_baseline_stats.dat', 'a')
f.write("\n-----\n" + 'x_feature_file = ' + x_file_name)
f.write("\n" + 'y_classification_file = ' + y_file_name)
f.write("\n" + 'data size = ' + str(len(x)) )
f.write("\n" + 'train_accuracy = ' + str(float(train_num_correct) / split_index))
f.write("\n" + 'test_accuracy = ' + str(float(test_num_correct) / (len(x) - split_index)))
f.close()
