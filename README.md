dependencies - 
mlpy

data -
i.stanford.edu/~adityaj/cs229_fall2014_dataset.html
Add amazon_products and AmazonHierarchy.json to data/

./stats/correlation_mat.npy is a 31x31 row normalized correlation matrix of all top level book labels for about 100,000 book ite

Task 0 - 
Build tools to extract feature vector, tf-idf values, label to index mapping
for level 0 labels

Task 1 - 
Extract level 1 labels of main category "Books" using book_hierarchy_parser.py
and run softmax regression over all instances of Books in amazon_products using book_data_parser.py

Task 2 - 
Creates Tf-Idf input vector for each review for Amazon Products

Task 3 - 

Task 4 - 

Task 5 - 

Task 6 -
Generates normalized correlation matrix and generates stats using current correlation algorithm and uncorrelated algorithm