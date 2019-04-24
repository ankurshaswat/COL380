import sys
import re

path = sys.argv[1]

with open(path) as f:
    x = f.readlines()

text_len = int(x[0].strip().split()[0])
num_patterns = int(x[0].strip().split()[1])

text = x[1].strip()

lens = x[2].strip().split()
periods = x[3].strip().split()

# print(x)

for i in range(num_patterns):
    pattern = x[4+i].strip()
    # print(pattern)
    # print(text)
    print([m.start() for m in re.finditer('(?='+pattern+')', text)])


# print(text_len,num_patterns,text)