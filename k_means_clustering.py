"""
Source: Ella
"""

def grouped(group1_centroid, group2_centroid, data):
    group1_index = []
    group2_index = []
    group1 = []
    group2 = []
    sum1 = 0
    sum2 = 0
    
    for i in range(len(data)):
        dist_1 = abs(data[i] - group1_centroid)
        dist_2 = abs(data[i] - group2_centroid)
        
        if dist_1 < dist_2:
            group1_index.append(i + 1)
            group1.append(data[i])
            sum1 += data[i]
        else:
            group2_index.append(i + 1)
            group2.append(data[i])
            sum2 += data[i]
    
    print("group 1 (index): ")
    print(group1_index)
    print("group 1 (element): ")
    print(group1)
    print("total assigned to group 1: " + str(len(group1)))
    print()
    print("group 2 (index): ")
    print(group2_index)
    print("group 2 (element): ")
    print(group2)
    print("total assigned to group 2: " + str(len(group2)))
    
    new_centroid1 = sum1 / len(group1)
    new_centroid2 = sum2 / len(group2)
    print()
    print("new centroid 1: " + str(new_centroid1))
    print("new centroid 2: " + str(new_centroid2))
    

data = [50, 60, 66, 68, 71, 72, 75, 82, 90, 99]

group1_centroid = 66
group2_centroid = 75

grouped(group1_centroid, group2_centroid, data)