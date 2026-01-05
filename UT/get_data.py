import numpy as np
import os
import os.path
import random

def read_file(file_path, number_array, number_array_batch):
    data = np.genfromtxt(file_path, delimiter=',')
    data = data[1:, 2:]
    beam_num = number_array - number_array_batch + 1
    frame_num = int(data.size / (beam_num * data.shape[1]))
    
    cscan_data = np.zeros((beam_num, frame_num, data.shape[1]))
    for j in range(frame_num):
        cscan_data[:, j, :] = data[j*beam_num:(j+1)*beam_num, :]

    return cscan_data


data_dir = os.getcwd()
data_dir = data_dir.replace('Codes', 'Data')

number_array = 128  # 128 for 0823 & 0521, 64 for others
number_array_batch = 4

date = '0710_A'
depth = 274  # 320 for 1128, 344 for 0805 & 0806, 267 for 0813 & 0821 & 0823, 251 for 1122, 274 for 0521 & 0710

whether_CV = False
whether_CC = True


# 0805
# nbeam_L = [16, 10, 12, 13, 15, 15, 15, 12, 14, 15, 14, 15, 13,  9,  6,  6, 10, 10,  9, 11, 10]
# nbeam_U = [48, 40, 41, 43, 41, 45, 44, 40, 40, 43, 43, 40, 41, 46, 44, 48, 43, 41, 44, 43, 46]
# nscan_L = [22, 20, 24, 23, 22, 24, 22, 22, 22, 22, 23, 20, 20, 20, 17, 20, 17, 17, 17, 17, 20]
# nscan_U = [43, 44, 44, 46, 46, 46, 44, 46, 45, 44, 45, 38, 40, 46, 45, 49, 46, 47, 44, 45, 43]

# 0806
# nbeam_L = [10, 12,  14, 15, 18, 17, 11,  9]
# nbeam_U = [43, 42,  43, 41, 44, 46, 44, 43]
# nscan_L = [40, 30,  50, 30, 35, 35, 40, 35]
# nscan_U = [91, 81, 101, 81, 86, 86, 91, 91]

# 0813
# nbeam_L = [10, 10, 13, 17, 10, 20, 16, 19, 12, 10, 10,  7, 13, 14,  9, 12, 10, 11,  8, 10, 12, 10,  8, 10,  7,  9, 10, 10,  5,  6,  4,  7,  6,  5,  5,  7]
# nbeam_U = [43, 39, 43, 44, 43, 45, 42, 45, 41, 41, 43, 40, 40, 44, 43, 42, 43, 41, 42, 44, 46, 45, 43, 43, 42, 44, 41, 43, 48, 48, 48, 45, 41, 44, 43, 45]
# nscan_L = [22, 18, 23, 16, 18, 15, 18, 26, 25, 17, 23, 24, 23, 20, 20, 20, 20, 30, 20, 32, 20, 20, 20, 20, 24, 20, 19, 19, 20, 15, 20, 18, 18, 17, 15, 14]
# nscan_U = [43, 41, 46, 38, 44, 41, 43, 51, 47, 43, 47, 50, 46, 44, 45, 44, 43, 51, 46, 56, 48, 47, 48, 48, 51, 48, 46, 49, 53, 51, 55, 51, 51, 48, 46, 43]

# 0821
# nbeam_L = [10, 10, 13, 17, 10, 20, 16, 19, 12, 10,  8, 10,  9, 10, 10, 12, 10, 10,  7, 13, 14,  9, 12, 10, 11,  8, 10,  7, 13, 10, 10, 12,  5,  6,  4,  7,  6,  5,  5,  7]
# nbeam_U = [43, 39, 43, 44, 43, 45, 42, 45, 46, 45, 43, 43, 44, 41, 43, 41, 41, 43, 40, 40, 44, 43, 42, 43, 41, 42, 44, 42, 44, 43, 39, 42, 48, 48, 48, 45, 41, 44, 43, 45]
# nscan_L = [22, 18, 23, 16, 18, 15, 18, 26, 20, 20, 20, 20, 20, 19, 19, 25, 17, 23, 24, 23, 20, 20, 20, 20, 30, 20, 32, 24, 20, 25, 20, 25, 20, 15, 20, 18, 18, 17, 15, 14]
# nscan_U = [43, 41, 46, 38, 44, 41, 43, 51, 48, 47, 48, 48, 48, 46, 49, 47, 43, 47, 50, 46, 44, 45, 44, 43, 51, 46, 56, 51, 46, 51, 46, 51, 53, 51, 55, 51, 51, 48, 46, 43]

# 0823:     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39
# nbeam_L = [49, 56, 49, 48, 49, 48, 47, 43, 51, 49, 48, 48, 50, 50, 45, 49, 46, 48, 47, 46, 44, 43, 43, 47, 45, 44, 48, 47, 50, 50, 51, 47, 47, 41, 43, 50, 47, 46, 41, 47]
# nbeam_U = [75, 81, 73, 72, 75, 70, 73, 67, 78, 77, 73, 73, 76, 73, 70, 72, 70, 73, 71, 64, 65, 66, 64, 72, 70, 72, 73, 75, 76, 77, 77, 68, 73, 71, 73, 77, 72, 72, 69, 75]
# nscan_L = [33, 31, 31, 32, 42, 43, 38, 33, 35, 34, 35, 33, 32, 33, 31, 33, 27, 45, 29, 31, 34, 34, 30, 32, 30, 32, 31, 39, 37, 41, 34, 32, 23, 29, 30, 40, 29, 35, 31, 30]
# nscan_U = [54, 51, 52, 53, 64, 67, 59, 56, 56, 56, 59, 56, 53, 51, 49, 52, 45, 64, 48, 49, 52, 54, 50, 53, 50, 52, 52, 58, 56, 60, 53, 51, 48, 58, 53, 67, 53, 55, 52, 49]

# 1122:     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61  62  63
# nbeam_L = [12, 12,  6, 12, 12, 19, 16,  8, 21, 11, 13, 11, 12, 16, 14, 15, 21, 16, 10, 14, 19, 17, 15, 14, 14, 15, 13, 13, 12, 11, 12, 15, 16, 17, 13, 15, 11, 16, 17, 14, 14, 16, 17, 13, 16, 16, 14, 15, 15, 17, 15, 14, 14, 13, 12, 16, 11,  6,  7,  9, 10,  8,  8,  7]
# nbeam_U = [39, 41, 34, 37, 39, 42, 42, 35, 46, 38, 39, 40, 37, 41, 39, 40, 44, 42, 43, 42, 41, 41, 42, 42, 41, 42, 40, 40, 39, 38, 39, 46, 41, 42, 38, 40, 36, 41, 45, 45, 42, 42, 46, 44, 43, 45, 41, 42, 42, 41, 41, 41, 40, 40, 39, 41, 47, 47, 49, 45, 42, 42, 43, 41]
# nscan_L = [23, 15, 19, 19, 22, 19, 24, 17, 26, 20, 22, 20, 22, 20, 23, 20, 20, 20, 18, 21, 23, 22, 22, 23, 26, 22, 22, 22, 38, 25, 22, 22, 26, 22, 26, 23, 30, 25, 37, 24, 23, 37, 21, 20, 20, 19, 18, 19, 19, 18, 19, 25, 22, 28, 21, 22, 19, 15, 15, 13, 15, 16, 12, 14]
# nscan_U = [46, 38, 40, 40, 45, 40, 49, 40, 50, 45, 47, 48, 47, 45, 46, 43, 41, 43, 49, 47, 45, 42, 48, 50, 53, 49, 48, 49, 69, 50, 49, 50, 51, 45, 48, 48, 53, 47, 64, 52, 47, 58, 46, 44, 42, 43, 43, 43, 42, 40, 41, 48, 43, 50, 45, 44, 51, 50, 51, 43, 44, 45, 40, 44]

# 0210:     0   1   2   3   4   5  | Only panel
# nbeam_L = [ 9, 11, 15, 11, 10, 14]
# nbeam_U = [38, 44, 42, 43, 42, 45]
# nscan_L = [ 2,  1,  2,  2,  2,  2]
# nscan_U = [70, 55, 66, 72, 64, 59]

# 0521:     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29  30  31  32  33  34  35  36  37  38  39  40  41  42  43  44  45  46  47  48  49  50  51  52  53  54  55  56  57  58  59  60  61
# nbeam_L = [48, 50, 47, 48, 50, 57, 48, 49, 48, 55, 53, 50, 48, 49, 52, 48, 49, 54, 52, 54, 51, 50, 52, 48, 50, 53, 58, 47, 53, 50, 52, 44, 47, 50, 48, 50, 43, 45, 52, 48, 50, 56, 49, 48, 52, 44, 49, 50, 51, 55, 57, 54, 57, 52, 50, 47, 50, 46, 50, 48, 52, 43]
# heights = [23, 22, 22, 20, 23, 23, 24, 23, 24, 26, 19, 21, 22, 21, 26, 24, 28, 25, 20, 20, 24, 24, 23, 23, 26, 24, 22, 23, 25, 24, 22, 24, 21, 20, 26, 25, 27, 28, 22, 24, 27, 26, 25, 27, 25, 26, 25, 21, 27, 20, 22, 19, 18, 27, 28, 30, 25, 22, 25, 26, 27, 28]
# nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
# nscan_L = [37, 35, 33, 33, 38, 40, 45, 40, 42, 42, 42, 42, 43, 43, 44, 35, 42, 55, 33, 39, 40, 39, 44, 40, 41, 56, 42, 35, 40, 29, 40, 42, 40, 35, 38, 36, 39, 40, 45, 55, 39, 39, 38, 38, 45, 40, 40, 51, 49, 39, 37, 42, 59, 39, 32, 37, 39, 50, 36, 40, 47, 38]
# widths  = [24, 22, 20, 24, 24, 24, 24, 20, 23, 22, 23, 24, 24, 26, 24, 20, 26, 26, 18, 20, 23, 24, 23, 26, 22, 24, 21, 18, 25, 24, 21, 20, 20, 18, 21, 19, 26, 24, 17, 18, 23, 20, 22, 23, 23, 19, 23, 19, 22, 19, 23, 23, 21, 30, 26, 23, 27, 27, 24, 23, 23, 25]
# nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]

# 0710_AB:  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24  25  26  27  28  29
# nbeam_L = [48, 49, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 53, 46, 43, 52, 53, 51, 54, 53, 52, 47, 52, 54, 53]
# heights = [22, 27, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 26, 27, 26, 26, 26, 27, 27, 25, 26, 26, 26, 26, 27]
# nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
# nscan_L = [46, 41, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 45, 43, 36, 46, 47, 38, 40, 41, 42, 39, 58, 59, 36]
# widths  = [23, 21, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 27, 27, 29, 28, 23, 27, 25, 24, 25, 28, 29, 28, 28]
# nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]

# 0710_A:   0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
nbeam_L = [48, 49, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 53, 46, 43]
heights = [22, 27, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 26, 27, 26]
nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
nscan_L = [46, 41, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 45, 43, 36]
widths  = [23, 21, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 27, 27, 29]
nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]

# 0710_A_wo2&18:   0   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  18  19
# nbeam_L =        [48, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 46, 43]
# heights =        [22, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 27, 26]
# nbeam_U =        [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
# nscan_L =        [46, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 43, 36]
# widths  =        [23, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 27, 29]
# nscan_U =        [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]

# 0802:     0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19
# nbeam_L = [48, 50, 49, 49, 46, 42, 48, 45, 46, 50, 49, 49, 52, 51, 53, 49, 43, 53, 46, 43]
# heights = [22, 28, 26, 28, 31, 25, 26, 25, 27, 25, 26, 27, 31, 25, 25, 24, 26, 26, 27, 26]
# nbeam_U = [nbeam_L[i] + heights[i] + 1 for i in range(len(nbeam_L))]
# nscan_L = [46, 40, 29, 36, 41, 35, 46, 36, 46, 41, 64, 45, 50, 41, 40, 41, 41, 38, 43, 36]
# widths  = [23, 20, 31, 26, 29, 23, 26, 27, 29, 25, 29, 31, 27, 21, 28, 22, 27, 22, 27, 29]
# nscan_U = [nscan_L[i] + widths[i] + 1 for i in range(len(nscan_L))]


sample_num = len(nbeam_L)
if whether_CV:
    training_index = list(range(10, 20))
    for cv_index in range(10, 20):
        training_indices = [i for i in training_index if i != cv_index]
        testing_indices = list(set(list(range(20))) - set(training_indices))

        ascan_pb = np.zeros((100000, depth))
        ascan_train = np.zeros((100000, depth))
        ascan_test = np.zeros((100000, depth))
        ascan_panel = np.zeros((100000, depth))
        ascan_all = np.zeros((100000, depth))

        start_temp = 0
        for k in range(sample_num):
            if k < 10:
                data_filename = '00' + str(k) + '.csv'
            else:
                data_filename = '0' + str(k) + '.csv'
            data_file_path = os.path.join(data_dir + '\\2025' + date, data_filename)
            cscan_data = read_file(data_file_path, number_array, number_array_batch)
            # cscan_data = (cscan_data - 127.5) / 127.5  # Normalization

            # Cropping
            nbeam = nbeam_U[k] - nbeam_L[k]
            nscan = nscan_U[k] - nscan_L[k]
            cscan_bond = cscan_data[nbeam_L[k]:nbeam_U[k], nscan_L[k]:nscan_U[k], :].reshape((nbeam*nscan, depth))
            
            if k in training_indices:
                ascan_train[start_temp:(start_temp+nbeam*nscan), :] = cscan_bond
            elif k in testing_indices:
                ascan_test[start_temp:(start_temp+nbeam*nscan), :] = cscan_bond
            ascan_all[start_temp:(start_temp+nbeam*nscan), :] = cscan_bond

            start_temp += cscan_bond.shape[0]

        ascan_train = ascan_train[~np.all(ascan_train == 0, axis=1)]
        ascan_test = ascan_test[~np.all(ascan_test == 0, axis=1)]
        ascan_all = ascan_all[~np.all(ascan_all == 0, axis=1)]

        np.savetxt('ascan_train_' + date + '_' + str(cv_index+1) + '.csv', ascan_train, delimiter = ',')
        np.savetxt('ascan_test_' + date + '_' + str(cv_index+1) + '.csv', ascan_test, delimiter = ',')
elif whether_CC: 
    ascan_pb = np.zeros((100000, depth))
    ascan_train = np.zeros((100000, depth))
    ascan_test = np.zeros((100000, depth))
    ascan_panel = np.zeros((100000, depth))
    ascan_all = np.zeros((100000, depth))

    training_indices = list(range(10, 20))
    testing_indices = list(range(10))
    selected = random.sample(training_indices, 3)
    # selected = [12, 17, 19]
    for idx in selected:
        training_indices.remove(idx)
    testing_indices.extend(selected)

    start_temp = 0
    for k in range(sample_num):
        if k < 10:
            data_filename = '00' + str(k) + '.csv'
        else:
            data_filename = '0' + str(k) + '.csv'
        data_file_path = os.path.join(data_dir + '\\2025' + date, data_filename)
        cscan_data = read_file(data_file_path, number_array, number_array_batch)

        # Cropping
        nbeam = nbeam_U[k] - nbeam_L[k]
        nscan = nscan_U[k] - nscan_L[k]
        cscan_bond = cscan_data[nbeam_L[k]:nbeam_U[k], nscan_L[k]:nscan_U[k], :].reshape((nbeam*nscan, depth))
            
        if k in training_indices:
            ascan_train[start_temp:(start_temp+nbeam*nscan), :] = cscan_bond
        elif k in testing_indices:
            ascan_test[start_temp:(start_temp+nbeam*nscan), :] = cscan_bond
        ascan_all[start_temp:(start_temp+nbeam*nscan), :] = cscan_bond

        start_temp += cscan_bond.shape[0]

    ascan_train = ascan_train[~np.all(ascan_train == 0, axis=1)]
    ascan_test = ascan_test[~np.all(ascan_test == 0, axis=1)]
    ascan_all = ascan_all[~np.all(ascan_all == 0, axis=1)]

    file_name_suffix = date + '_' + str(selected[0]-9) + '_' + str(selected[1]-9) + '_'+ str(selected[2]-9)
    print(file_name_suffix)
    np.savetxt('data\\ascan_train_' + file_name_suffix + '.csv', ascan_train, delimiter = ',')
    np.savetxt('data\\ascan_test_' + file_name_suffix + '.csv', ascan_test, delimiter = ',')

# np.savetxt('data\\ascan_all_' + file_name_suffix + '.csv', ascan_all, delimiter = ',')

'''
ascan_pb -= 127
ascan_pb_norm = np.zeros(ascan_pb.shape)
for i, ascan in enumerate(ascan_pb):
    ascan_pb_norm[i, :] = ascan / np.max(ascan[0:100])

ascan_all -= 127
ascan_all_norm = np.zeros(ascan_all.shape)
for i, ascan in enumerate(ascan_all):
    ascan_all_norm[i, :] = ascan / np.max(ascan[0:100])

np.savetxt('data\\ascan_pb_norm_' + date + '.csv', ascan_pb_norm, delimiter = ',')
np.savetxt('data\\ascan_all_norm_' + date + '.csv', ascan_all_norm, delimiter = ',')
'''
