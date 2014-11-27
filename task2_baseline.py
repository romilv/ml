import json
import mlpy
import math
import task3_normalize_y
import numpy

DATA_SIZE = '500'
x_file_name = './data/task2_tfidf2d_list' + DATA_SIZE + '.json'
x_file_data = open(x_file_name).read()
x = json.loads(x_file_data)

y_file_name = './data/task2_y' + DATA_SIZE + '.json'
y_trimmed = task3_normalize_y.normalize_y(y_file_name)

numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json')))
# arr = [-1]*len(x)
# y = numpy.array([list(arr) for _ in xrange(numLabels)])

y_open = open(y_file_name)
y_data = numpy.array(json.loads(y_open.read()))
y_open.close()
# for i,label_list in enumerate(y_data):
#     for label in label_list:
#         y[label][i]=1


# we use 70% of x for training and 30% for testing
TRAIN_PERCENT = 0.7
split_index = int(math.floor(TRAIN_PERCENT * len(x)))

model = mlpy.LibLinear(solver_type='l2r_l2loss_svc')
model.learn(x[:split_index], y_trimmed[:split_index])


# num_labels = max(y_trimmed)

# train_confusion = [[0 for _ in range(num_labels)] for _ in range(num_labels)]
# test_confusion = [[0 for _ in range(num_labels)] for _ in range(num_labels)]

def error(pred_list, actual_set, n):
    # n >= 1
    correct= 0.0
    if len(pred_list) != 0:
        for elem in pred_list:
            if elem  in actual_set:
                correct += 1.0
        # ~ fraction match in actual labels time ~ fraction overmatched in predicted labels
        return 1-(correct/len(actual_set)) * math.pow((correct/len(pred_list)), 1.0/n)
    else:
        return 1

# train_num_correct = 0
# test_num_correct = 0
train_error, test_error = 0, 0
# train error
for i in range(split_index):
    prediction = [model.pred(x[i])]
    # actual = y[i]
    actual_labels = set(y_data[i])
    train_error += error(prediction, actual_labels, 3.0)
    # if prediction == actual:
        # train_num_correct += 1
    # train_confusion[actual - 1][prediction - 1] += 1

train_error = train_error/split_index

# test error
for i in range(split_index, len(x)):
    prediction = [model.pred(x[i])]
    actual_labels = set(y_data[i])
    test_error += error(prediction, actual_labels, 3.0)
    # actual = y_trimmed[i]
    # if prediction == actual:
        # test_num_correct += 1
    # test_confusion[actual - 1][prediction - 1] += 1
test_error = test_error/(len(x)-split_index)
# def print_matrix(matrix):
#     for row in matrix:
#         print row

# print "=========================================="
# print_matrix(train_confusion)
# print "=========================================="
# print_matrix(test_confusion)
# print "=========================================="

# print train_num_correct, "/", split_index
# print test_num_correct, "/", len(x) - split_index

# with open('./data/task2_train_confusion.json', 'w') as outfile:
#     json.dump(train_confusion, outfile)
# outfile.close()

# with open('./data/task2_test_confusion.json', 'w') as outfile:
#     json.dump(test_confusion, outfile)
# outfile.close()

f = open('./stats/task7_baseline_stats.dat', 'a')
f.write("\n--------\n" + 'x_feature_file = ' + x_file_name)
f.write("\n" + 'y_classification_file = ' + y_file_name)
f.write("\n" + 'DATA_SIZE = ' + str(len(x)) )
f.write("\n" + 'split_index = ' + str(TRAIN_PERCENT))
# f.write("\n" + 'train_accuracy = ' + str(float(train_num_correct) / split_index))
f.write("\n" + 'train_error = ' + str(train_error))
f.write("\n" + 'test_error = ' + str(test_error))
# f.write("\n" + 'test_accuracy = ' + str(float(test_num_correct) / (len(x) - split_index)))
f.close()
