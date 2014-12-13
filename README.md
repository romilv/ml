Commands - 
python task0_extract_book_data.py
python task2_amazon_book_data_parser.py
python task3_social_booK_data_parser.py
python task6_generate_correlation_mat.py
python task6_generate_social_correlation_mat.py
python task6_multilabel.py
python task7_multilabel_heuristic.py
python task13_multilabel_NB.py
python task13_multilabel_RF.py
python task12_kfold.py




dependencies - 
numpy
mlpy
scikit-learn



data -
i.stanford.edu/~adityaj/cs229_fall2014_dataset.html
correlation_mat.npy is a 31x31 row normalized correlation matrix of all top level book labels for about 100,000 book ite



Task 0 - 
Build tools to extract feature vector, tf-idf values, label to index mapping
for level 0 labels

Task 1 - 
Extract level 1 labels of main category "Books" using book_hierarchy_parser.py
and run softmax regression over all instances of Books in amazon_products using book_data_parser.py

Task 2 - 
Creates Tf-Idf input vector for each review for Amazon Products

Task 3 - 
Selects a single label for each review to attempt a uniform distribution of output labels
Also trains on the Amazon dataset and tests on the Twitter dataset

Task 4 - 
Predicts multiple labels after training on Amazon and testing on Amazon and after training on Amazon and testing on Twitter

Task 5 - 
Predicts multiple labels using a correlation matrix calculated using the training data

Task 6 -
Generates normalized correlation matrix and generates stats using current correlation algorithm and uncorrelated algorithm. Final algorithm using top J independent labels followed by top K tuples

Task 7 - 
Multilabel prediction using heuristics to estimate the correct number of labels to predict

Task 8 - 
Multilabel prediction using 31 choose 2 correlation models for every pair of labels instead of a fixed correlation matrix

svd_reduce.py - 
Reduces feature vectors using singular value decomposition

Task 10 - 
Same as task 8, but using feature vectors reduced by singular value decomposition

Task 11 - 
Varies hyper-paramaters (not gamma) within specified ranges, finding the values producing the lowest error of all combinations (we tested 1760 hyper-parameter combinations)

Task 12 - 
Performs K-fold validation

Task 13 - 
Evaluates on Naive Bayes and Random Forest

util.py - 
Stemming algorithm


-----------------------------


Evaluation score for data.zip

For 3000 reviews from Amazon (task2_tfidf2d_list3000.json, task2_y3000.json)
Running task6_multilabel.py

Baseline algorithm
Train on Amazon Error - 0.074786841%
Test on Amazon Error - 55.2977788%
Precision - 0.6724
Recall - 0.4789
F1 - 0.5594
FBeta (Beta = 1/3) - 0.4932


Correlation Algorithm
Train on Amazon Error - 15.82623588%
Test on Amazon Error - 42.40562757%
Precision - 0.4644
Recall - 0.6756
F1 - 0.5504
FBeta (Beta = 1/3) - 0.6421


For 3000 reviews from Twitter (task3_social_tfidf2d_list3000.json, task3_social_y3000.json)
running task7_multilabel_heuristic.py

Baseline algorithm
Train on Twitter Error - 0.074786841%
Test on Twitter Error - 37.10096959%
Precision - 0.7464
Recall - 0.6308
F1 - 0.6837
FBeta (Beta = 1/3) - 0.6407

Correlation Algorithm
Train on Twitter Error - 15.82623588%
Test on Twitter Error - 37.13123203%
Precision - 0.7219
Recall - 0.6306
F1 - 0.6733
FBeta (Beta = 1/3) - 0.6388

Link to code repository online https://github.com/vrma/ml