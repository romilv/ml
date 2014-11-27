from nltk.stem import PorterStemmer


def stem_array(words_arr):
	st = PorterStemmer()
	stemmed_arr = []
	for word in words_arr:
		stemmed_arr.append(st.stem(word))
	return stemmed_arr