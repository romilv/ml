import json
import mlpy
import numpy
import math
import itertools
import heapq
from sklearn.ensemble import RandomForestClassifier

STATS_FILE = './stats/task13_amzn_stats_RF.dat'

with open(STATS_FILE, 'a') as f:
    f.write('\n\n\n\n--------------------------------')
f.close()

for size_iter in range(1000, 1001, 500):
    SIZE = str(size_iter)

    # AMAZON TO AMAZON
    SOLVER = 'Random Forest Classifier' 
    ALPHA = 0.3
    N = 3 # 1/importance given to overprediction
    K = 6 # top K pairs from correlated probabilities
    J = 12
    TRAIN_PERCENT = 0.7 # split index

    with open(STATS_FILE, 'a') as f:
        f.write('\n\n--------')
        f.write('\nSIZE ' + SIZE)
        f.write('\nalpha ' + str(ALPHA) + '\nJ ' + str(J) + '\nK ' + str(K))
    f.close()

##    x_file_name = './data/task3_social_tfidf2d_list' + SIZE + '.json' # twitter
    x_file_name = './data/task2_tfidf2d_list' + SIZE + '.json' # amazon
    x_file_data = open(x_file_name).read()
    x = numpy.array(json.loads(x_file_data))

    y_file_name = './data/task2_y' + SIZE + '.json' # amazon
##    y_file_name = './data/task3_social_y' + SIZE + '.json' # twitter

    numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json'))) # get number of labels
    # generate y as a 2D array where we indicate presence/absence of each label
    #y is column major
    arr = [-1]*len(x)
    y = numpy.array([list(arr) for _ in xrange(numLabels)])

    corr_y = numpy.load('./stats/correlation_mat.npy') # amazon
##    corr_y = numpy.load('./stats/social_correlation_mat.npy') # twitter

    y_open = open(y_file_name)
    y_data = numpy.array(json.loads(y_open.read()))
    y_open.close()
    for i,label_list in enumerate(y_data):
        for label in label_list:
            y[label][i]=1

    split_index = int(math.floor(TRAIN_PERCENT * len(x)))

    #Train model for all labels
    models = []
    for label in y:
##        temp = mlpy.LibLinear(solver_type=SOLVER, C=C)
        temp = RandomForestClassifier()
        temp.fit(x[:split_index],label[:split_index])
        models.append(temp)
        pass

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
        
    def evaluate_model(test_type, start, end, alpha):
        heap_err, pred_err = 0, 0
        for i in range(start,end):
##            temp_probs = [(1-model.pred_probability(x[i])[0], idx, model.labels()) for idx,model in enumerate(models)] 
            temp_probs = [(1-model.predict_proba(x[i])[0][0], idx, model.classes_) for idx,model in enumerate(models)] 
            probabilities = []

            for idx in xrange(len(temp_probs)):
                label = temp_probs[idx][2]
##                print temp_probs[idx]
##                print temp_probs[idx][0]
##                print '======================'
                if label[0] == 1:
                    probabilities.append((1 - temp_probs[idx][0], temp_probs[idx][1]))
                else:
                    probabilities.append((temp_probs[idx][0], temp_probs[idx][1]))
            
            predictions = [idx for idx,model in enumerate(models) if model.predict(x[i])==1 ]
##            print probabilities
            probabilities.sort(reverse=True)
            probabilities =  probabilities[:J] # consider 10 with highest Pr


            # considering tuples from correlation matrix and choosing top K=4
            h = [] # heap
            pairs = itertools.combinations(probabilities,2)
            for a,b in pairs:
                p_a_b = math.sqrt(corr_y[a[1]][b[1]] * corr_y[b[1]][a[1]])
                p_a_b = math.pow(p_a_b, alpha)

                pr = a[0]*b[0]*p_a_b
                key = (a[1], b[1])
                heapq.heappush(h, (pr, key))
            h1 = heapq.nlargest(K, h) # how many to choose in the end
            final_h = set()
            for elem in h1: # distinct
                final_h.add(elem[1][0])
                final_h.add(elem[1][1])
            final_h = list(final_h)

            actual_labels = set(y_data[i])

            cur_heap_err = error(final_h, actual_labels, N)
            cur_pred_err = error(predictions, actual_labels, N)
            heap_err += cur_heap_err
            pred_err += cur_pred_err

        with open(STATS_FILE, 'a') as f:
            f.write("\nerror type " + test_type)
            f.write('\nwith correlation (heap) ' + str(heap_err/(end-start)))
            f.write("\noriginal pred " + str(pred_err/(end-start)))
            f.write("\n--------")
        f.close()
        # end evaluate_model

    evaluate_model('TRAIN', 0, split_index, ALPHA)
    evaluate_model('TEST', split_index, len(x), ALPHA)
