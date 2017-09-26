# BABU RAO TALACHUTLA
# 700663195


from collections import Counter
import numpy as np
import operator
from sys import argv

#funtion to load our text files and create a data matrix and labels seperately!!!
def load_data(filename):
    input_file = open(filename)
    lines = input_file.readlines()
    data_matrix = np.zeros((len(lines),8))
    labels = []
    index = 0
    for line in lines:
        line = line.strip()
        tokens = line.split(',')
        data_matrix[index, :] = tokens[:8]
        labels.append(int(tokens[-1]))
        index += 1
    return data_matrix, labels

#function to normalize the data matrix and returns normalized data matrix
# Normalization requires for put all our data between 0 and 1
def normalize(X):
    min_vals = X.min(axis=0)
    max_vals = X.max(axis=0)
    ranges = max_vals - min_vals
    nrows = X.shape[0]
    norm_data_matrix = X - np.tile(min_vals, (nrows, 1))
    norm_data_matrix = norm_data_matrix / np.tile(ranges, (nrows, 1))
    return norm_data_matrix, ranges, min_vals


#kNN classifier; returns the classified labels for our data
def classify_kNN(test_x, X, Y, k):
    dataset_size = X.shape[0]
    diff_matrix = np.tile(test_x, (dataset_size, 1)) - X
    sq_diff_matrix = diff_matrix**2
    sq_distances = sq_diff_matrix.sum(axis=1)
    distances = sq_distances**0.5
    sorted_distances_indices = distances.argsort()
    label_counts = {}
    for i in range(k):
        neighbor_label = Y[sorted_distances_indices[i]]
        label_counts[neighbor_label] = \
            label_counts.get(neighbor_label, 0) + 1
    sorted_label_counts = sorted(label_counts.iteritems(),
                                 key=operator.itemgetter(1),
                                 reverse=True)
    return sorted_label_counts[0][0]

#Tests our trained data against test data and returns error count
def test_kNN(train_X, train_Y, test_X, test_Y, k):
    train_data_matrix, ranges, min_vals = normalize(train_X)
    test_data_matrix, text_ranges, text_min_vals = normalize(test_X)
    dataset_size = test_X.shape[0]
    error_count = 0.0
    for i in range(dataset_size):
        guess = classify_kNN(test_data_matrix[i, :],
                             train_data_matrix,
                             train_Y, k)
        if (guess != test_Y[i]):
            error_count += 1
    print '(k, error_count, |test|, error_rate): (%d, %d, %d, %f)' % (k, error_count, test_X.shape[0], (error_count / test_X.shape[0]))

#main method; takes command line arguments
def main(args):
    file1=argv[1]
    file2 = argv[2]
    train_X, train_Y = load_data(file1)
    test_X, test_Y = load_data(file2)
    k=int(argv[3])
    for i in range(1, k+1):
        test_kNN(train_X, train_Y, test_X, test_Y, i)

if __name__ == '__main__':
    import sys
    main(sys.argv[1:])

