import json
import mlpy
import math
import task3_normalize_y

x_amazon_file_data = open("./data/task2_tfidf2d_list.json").read()
x_amazon = json.loads(x_amazon_file_data)

x_twitter_file_data = open("./data/task3_social_tfidf2d_list.json").read()
x_twitter = json.loads(x_twitter_file_data)

# we only use the first label from each list for now
y_amazon = task3_normalize_y.normalize_y("./data/task2_y.json")
y_twitter = task3_normalize_y.normalize_y("./data/task3_social_y.json")

# incorporate tfidf from amazon into twitter training data
#for row in range(len(x_amazon)):
#    for col in range(len(x_amazon[row])):
#        x_amazon[row][col] *= x_twitter[row][col]

model = mlpy.LibLinear(solver_type='l2r_lr')
model.learn(x_amazon, y_amazon)

train_num_correct = 0
test_num_correct = 0

num_train_labels = max(y_amazon)
num_test_labels = max(y_twitter)

train_confusion = [[0 for _ in range(num_train_labels)] for _ in range(num_train_labels)]
test_confusion = [[0 for _ in range(num_test_labels)] for _ in range(num_test_labels)]

# train error
for i in range(len(y_amazon)):
    prediction = model.pred(x_amazon[i])
    actual = y_amazon[i]
    if prediction == actual:
        train_num_correct += 1
    train_confusion[actual - 1][prediction - 1] += 1


# test error
for i in range(len(y_twitter)):
    prediction = model.pred(x_twitter[i])
    actual = y_twitter[i]
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

# print train_num_correct, "/", len(y_amazon)
# print test_num_correct, "/", len(y_twitter)

with open('./data/task3_train_confusion.json', 'w') as outfile:
    json.dump(train_confusion, outfile)
outfile.close()

with open('./data/task3_test_confusion.json', 'w') as outfile:
    json.dump(test_confusion, outfile)
outfile.close()

f = open('./stats/task3_amzn_stats.dat', 'a')
f.write("\n" + 'train_accuracy = ' + str(float(train_num_correct) / len(y_amazon)))
f.write("\n" + 'test_accuracy = ' + str(float(test_num_correct) / len(y_twitter)))
f.close()
