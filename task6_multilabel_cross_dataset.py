import json
import mlpy
import numpy
import math
import itertools
import heapq

tp = 0
tn = 0
fp = 0
fn = 0
btp = 0
btn = 0
bfp = 0
bfn = 0

for size_iter in range(3000, 3001, 500):
    SIZE = str(size_iter)

    # AMAZON TO AMAZON
    SOLVER = 'l2r_l2loss_svc' # best l2r_l2loss_svc
    C = 1 # default is 1
    ALPHA = 0.35 # for sqrt(p(a,b)) to power alpha, between 0.3 and 0.4 gives best result
    N = 3 # 1/importance given to overprediction
    K = 4 # top K pairs from correlated probabilities
    TRAIN_PERCENT = 1 # split index

    with open('./stats/task6_social_stats.dat', 'a') as f:
        f.write('\n\n--------')
        f.write('\nSOLVER ' + SOLVER)
        f.write('\nC ' + str(C))
        f.write('\nAMAZON SIZE ' + SIZE)
        f.write('\nTWITTER SIZE 3000')
        f.write('\nALPHA without SQRT ' + str(ALPHA))
        f.write('\nERROR-N ' + str(N))
        f.write('\nPAIRS-CHOSEN-K ' + str(K))
    f.close()

    # x amazon
    x_file_name = './data/task2_tfidf2d_list' + SIZE + '.json'
    x_file_data = open(x_file_name).read()
    x_amzn = numpy.array(json.loads(x_file_data))

    numLabels = len(json.load(open('./data/task1_amazon_book_label_map.json'))) # get number of labels

    # correlation matrix
    arr = [-1]*len(x_amzn)
    y_amzn = numpy.array([list(arr) for _ in xrange(numLabels)])
    # corr_y = numpy.load('./stats/correlation_mat.npy') # load correlation matrix
    corr_y = numpy.load('./stats/social_correlation_mat.npy') # load social correlation matrix
    
    # y amazon
    y_file_name = './data/task2_y' + SIZE + '.json'
    y_open = open(y_file_name)
    y_amzn_data = numpy.array(json.loads(y_open.read()))
    y_open.close()

    for i,label_list in enumerate(y_amzn_data):
        for label in label_list:
            y_amzn[label][i]=1

    split_index = int(math.floor(TRAIN_PERCENT * len(x_amzn)))

    #Train model for all labels on amazon
    models = []
    for label in y_amzn:
        temp = mlpy.LibLinear(solver_type=SOLVER, C=C)
        temp.learn(x_amzn,label)
        models.append(temp)
        pass

    # social x
    x_file_name = './data/task3_social_tfidf2d_list3000.json'
    x_file_data = open(x_file_name).read()
    x_twtr = numpy.array(json.loads(x_file_data))

    # twitter y
    y_file_name = './data/task3_social_y3000.json'
    y_open = open(y_file_name)
    y_twtr_data = numpy.array(json.loads(y_open.read()))
    y_open.close()  


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
        
    def evaluate_model(test_type, x, y_data, start, end, alpha):
        heap_err, pred_err = 0, 0
        for i in range(start,end):
            temp_probs = [(1-model.pred_probability(x[i])[0], idx, model.labels()) for idx,model in enumerate(models)] 
            probabilities = []

            for idx in xrange(len(temp_probs)):
                label = temp_probs[idx][2]
                if label[0] == 1:
                    probabilities.append((1 - temp_probs[idx][0], temp_probs[idx][1]))
                else:
                    probabilities.append((temp_probs[idx][0], temp_probs[idx][1]))
            
            predictions = [idx for idx,model in enumerate(models) if model.pred(x[i])==1 ]
            probabilities.sort(reverse=True)
            probabilities =  probabilities[:10] # consider 10 with highest Pr

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
            # final_h = set()
            final_h = list()
            for elem in h1: # distinct
                if not elem[1][0] in final_h:
                    final_h.append(elem[1][0])
                if not elem[1][1] in final_h:  
                    final_h.append(elem[1][1])
            num_baseline = len(predictions)
            before_final_h = final_h
            final_h = list(before_final_h[:num_baseline])
            # final_h = list(final_h)


            actual_labels = set(y_data[i])
            base_labels = set(predictions)
            predicts_labels = set(final_h)

            cur_heap_err = error(final_h, actual_labels, N)
            cur_pred_err = error(predictions, actual_labels, N)
            heap_err += cur_heap_err
            pred_err += cur_pred_err

            if test_type == 'TEST TWITTER 3000':
                global tp, tn, fp, fn,btp, btn, bfp, bfn
                tp += len(predicts_labels.intersection(actual_labels))
                fp += len(predicts_labels.difference(actual_labels))
                fn += len(actual_labels.difference(predicts_labels))
                tn += (31 - len(actual_labels.union(predicts_labels)))
                btp += len(base_labels.intersection(actual_labels))
                bfp += len(base_labels.difference(actual_labels))
                bfn += len(actual_labels.difference(base_labels))
                btn += (31 - len(actual_labels.union(base_labels)))

        with open('./stats/task6_social_stats.dat', 'a') as f:
            f.write('\n--------')
            f.write("\nerror type " + test_type)
            f.write('\nwith correlation (heap) ' + str(heap_err/(end-start)))
            f.write("\noriginal pred " + str(pred_err/(end-start)))
            f.write("\n--------")
        f.close()
        # end evaluate_model

    evaluate_model('TRAIN AMAZON', x_amzn, y_amzn_data, 0, len(x_amzn), ALPHA)
    evaluate_model('TEST TWITTER 3000', x_twtr, y_twtr_data, 0, len(x_twtr), ALPHA)

print 'tn', tn
print 'tp', tp
print 'fp', fp
print 'fn', fn
print'------------'
print 'btn', btn
print 'btp', btp
print 'bfp', bfp
print 'bfn', bfn