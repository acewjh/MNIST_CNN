import csv
import numpy as np

def load_data_set(file_path):
    with open(file_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        i = 0
        datalist = []
        for row in csvreader:
            if i == 0:  #skip the first line
                i += 1
                continue
            rowlist = [int(a) for a in row]
            datalist.append(rowlist)
            i += 1
    csvfile.close()
    data = np.array(datalist)
    labels = data[:, 0]
    train_set = data[:, 1:]
    return train_set, labels

def load_test_set(file_path):
    with open(file_path) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        i = 0
        datalist = []
        for row in csvreader:
            if i == 0:  #skip the first line
                i += 1
                continue
            rowlist = [int(a) for a in row]
            datalist.append(rowlist)
            i += 1
    csvfile.close()
    test_data = np.array(datalist)
    return test_data

def save_results(file_path, result):
    result_list = result.tolist()
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for i in range(len(result_list)):
            row = result_list[i]
            row.insert(0, i + 1)
            writer.writerow(row)
    csvfile.close()

# a test on save_results
# a = np.random.randn(128, 1)
# save_results('test_ret.csv', a)
