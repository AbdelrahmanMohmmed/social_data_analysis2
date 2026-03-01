file = open('Students_Data.txt')
read = file.readlines()
print(read)
text2 = []
for line in read:
    text2.append(line.strip())

print(text2)

students_data = []
for line in text2:
    student = dict(item.split(": ") for item in line.split(", "))
    students_data.append(student)

import pandas as pd
df = pd.DataFrame(students_data)
print(df.head())