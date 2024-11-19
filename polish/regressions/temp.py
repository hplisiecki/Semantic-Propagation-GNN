roberta_acc = [91.3,
93.9,
94.7,
97.3,
94.8,
96.1,
]

emoa_acc = [70.0,
66.8,
77.3,
73.7,
71.6,
84.3,
]

sprop_acc = [80.0,
88.9,
93.2,
92.2,
88.9,
90.7,
]

def compute_average_difference(a, b):
    return sum([a[i] - b[i] for i in range(len(a))]) / len(a)

print(compute_average_difference(roberta_acc, emoa_acc))
print(compute_average_difference(roberta_acc, sprop_acc))


roberta_prec = [86.7,
78.2,
71.2,
93.0,
81.4,
85.1,
]

emoa_prec = [70.1,
73.7,
39.6,
70.2,
52.3,
7.7,
]

sprop_prec = [65.9,
50.7,
72.1,
76.4,
61.5,
66.7,
]

print(compute_average_difference(roberta_prec, emoa_prec))
print(compute_average_difference(roberta_prec, sprop_prec))


roberta_recall = [
87.8,
63.0,
85.6,
90.7,
80.2,
86.4,
]

emoa_recall = [33.8,
19.9,
48.2,
47.5,
35.5,
18.0,
]

sprop_recall = [82.3,
35.1,
59.6,
77.8,
48.9,
65.1,
]

print(compute_average_difference(roberta_recall, emoa_recall))
print(compute_average_difference(roberta_recall, sprop_recall))