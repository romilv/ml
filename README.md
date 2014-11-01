dependencies - 
mlpy

data -
i.stanford.edu/~adityaj/cs229_fall2014_dataset.html
Add amazon_products and AmazonHierarchy.json to data/

Task 0 - 
Build tools to extract feature vector, tf-idf values, label to index mapping
for level 0 labels

Task 1 - 
Extract level 1 labels of main category "Books" using book_hierarchy_parser.py
and run softmax regression over all instances of Books in amazon_products using book_data_parser.py

TODO - 
Complete softmax regression implementation, stem data, get baseline results,
confusion matrix
