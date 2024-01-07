import torch 
import csv
import numpy as np

def write_pairwise_distances_to_csv(dat, output_file):
    with open(output_file, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        for i in range(dat.shape[0]):
            rowdata = [0]*(i+1)
            for j in range(i+1,dat.shape[0]):
                distance_value = np.array(torch.linalg.norm(dat[i]-dat[j])).astype(np.float16).item()
                rowdata.append(distance_value)
            csvwriter.writerow(rowdata)
            print(i)

write_pairwise_distances_to_csv(train_data, "E:\Data\HIsom_Data\CIFAR10_dmat.csv")