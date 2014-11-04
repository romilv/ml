import json
import mlpy
import math

x_file_data = open("./data/task2_tfidf2d_list.json").read()
x = json.loads(x_file_data)

y_file_data = open("./data/task2_y.json").read()
y = json.loads(y_file_data)

# we use 70% of x for training and 30% for testing
train_percent = 0.7
split_index = int(math.floor(train_percent * len(x)))

# we only use the first label from each list for now
y_trimmed = [label_list[0] for label_list in y]

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

print "=========================================="
print_matrix(train_confusion)
print "=========================================="
print_matrix(test_confusion)
print "=========================================="

print train_num_correct, "/", split_index
print test_num_correct, "/", len(x) - split_index

with open('./data/task2_train_confusion.json', 'w') as outfile:
    json.dump(train_confusion, outfile)
outfile.close()

with open('./data/task2_test_confusion.json', 'w') as outfile:
    json.dump(test_confusion, outfile)
outfile.close()

f = open('./data/task2_stats.dat', 'a')
f.write("\n" + 'train_accuracy = ' + str(float(train_num_correct) / split_index))
f.write("\n" + 'test_accuracy = ' + str(float(test_num_correct) / (len(x) - split_index)))
f.close()
