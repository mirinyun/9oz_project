import os
import csv
file_list = os.listdir("./test_data/u2net_outputs/A21")
print(file_list)

with open('A21_list.csv','w') as file :

    write = csv.writer(file)
    write.writerow(file_list)

list2 = []
for i in range (len(file_list)) :
    list2.insert(i,file_list[i].split(".")[0])
print(list2)


with open('A21_list2.csv','w') as file :

    write = csv.writer(file)
    write.writerow(list2)