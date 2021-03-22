import pickle
tablichka = pickle.load(open('/home/user/work/data_0.pkl','rb'))

table = []
for line_number,line in enumerate(tablichka):
    line_list = line.tolist()
    for i in range(0,len(line_list)):
        line_list[i][4] = line_number
        line_list[i].append(-1)
    table.append(line_list)

def get_routes(table):
    for i in range(0,len(table[0])):
        table[0][i][6] = i
    for line_number in range(1,len(table)):
        table[line_number-1], table[line_number] = find_roots(table[line_number-1], table[line_number])


def find_roots(root,line):
    for i, box in enumerate(line):
        for j, root_box in enumerate(root):
            root_rc = root_box[0]+root_box[2]
            box_rc = box[0] + box[2]
            if box_rc < root_box[0] or box[0] > root_rc:
                continue
            line[i][6].extend(root[j][6])
    return root, line
