import json
import mlpy
import numpy
import math
import itertools

print 'TRAIN ON AMZN'

# train on Amazon

x_amzn_file_name = './data/task2_tfidf2d_list.json'
x_amzn_file_data = open(x_amzn_file_name).read()
x = numpy.array(json.loads(x_amzn_file_data))

y_file_name = './data/task2_y.json'
#Get number of labels
numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json')))
#Generate y as a 2D array where we indicate presence/absence of each label
#y is column major
#Also build the correlation matrix
arr = [0]*len(x)
y = numpy.array([list(arr) for _ in xrange(numLabels)])
corr_y = numpy.array([list([0.0]*numLabels) for _ in xrange(numLabels)])
y_open = open(y_file_name)
y_data = numpy.array(json.loads(y_open.read()))
for i,label_list in enumerate(y_data):
    for label in label_list:
        y[label][i]=1
    for item in itertools.permutations(label_list,2):
        corr_y[item[0]][item[1]]+=1
    pass

# Normalize the correlation matrix
for idx,label in enumerate(corr_y):
    total = sum(label)
    if total==0:
        continue
    corr_y[idx]*=1.0/total


TRAIN_PERCENT = 1
split_index = int(math.floor(TRAIN_PERCENT * len(x)))

#Train model for all labels
models = []
for label in y:
    temp = mlpy.LibLinear(solver_type='l2r_lr')
    temp.learn(x[:split_index],label[:split_index])
    models.append(temp)
    pass

#Currently only try to predict two labels

#Train Error: Defined as 1 - fraction of predicted labels that were correct.
#Eg. If [1,2,3] is the set of labels and we predicted [3,4], then error is
#1-0.5=0.5
train_err=0
for i in range(split_index):
    predictions = [idx for idx,model in enumerate(models) if model.pred(x[i])==1 ]
    actual = set(y_data[i])
    if not predictions: train_err+=1.0
    elif len(predictions)==1:
        train_err += 0.5 if predictions[0] in actual else 1.0
    else:
        pairs = itertools.permutations(predictions,2)
        temp = [(corr_y[p[0]][p[1]],p) for p in pairs]
        predict2 = max(temp)[1]
        err = sum([0.5 for label in predict2 if label not in actual])
        train_err+=err
train_err/=split_index

#Test Error on Twitter
x_twtr_file_name = './data/task3_social_tfidf2d_list.json'
x_twtr_file_data = open(x_twtr_file_name).read()
x = numpy.array(json.loads(x_twtr_file_data))

y_file_name = './data/task3_social_y.json'
y_open = open(y_file_name)
y_data = numpy.array(json.loads(y_open.read()))

print "TEST ON TWTR"
test_err=0

for i in range(0,len(x)):
    predictions = [idx for idx,model in enumerate(models) if model.pred(x[i])==1 ]
    actual = set(y_data[i])
    if not predictions: test_err+=1.0
    elif len(predictions)==1:
        test_err += 0.5 if predictions[0] in actual else 1.0
    else:
        pairs = itertools.permutations(predictions,2)
        temp = [(corr_y[p[0]][p[1]],p) for p in pairs]
        predict2 = max(temp)[1]
        err = sum([0.5 for label in predict2 if label not in actual])
        test_err+=err
test_err/=(len(x))

f = open('./data/task4_amzn_twtr_correlation_mat.dat', 'w')
for item in corr_y:
    f.write('\t'.join([str(z) for z in item]).expandtabs(4))
f.close()

f = open('./stats/task4_amzn_twtr_stats.dat', 'a')
f.write('Data size : ' + str(len(x)) )
f.write("\n" + 'Training Error : ' + str(train_err*100))
f.write("\n" + 'Test Error : ' + str(test_err*100))
f.write("\n\n")
f.close()
