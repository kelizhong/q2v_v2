# coding=utf-8
from utils.data_util import sentence_gen
from utils.cache_util import RandomSet
from utils.data_util import aksis_data_label
import glob
import random
from itertools import combinations
import codecs


capacity = 65535
neg_number = 2
query_set = RandomSet()
res = list()

files = glob.glob("/Users/keliz/Downloads/aksis.purchased.pair/part*")
for num, line in enumerate(sentence_gen(files)):
    if num % 10000 == 0:
        print(num)
    items = line.split("\t")[1:]

    i = 0
    while True:
        i += 1
        query = random.choice(items)
        if len(query) > 2:
            query_set.add(query)
            break
        if i > len(items):
            break


fo = open("foo.txt", "w")
fo = codecs.open("foo.txt", "w", "utf-8")

finished_query = set()
for num, line in enumerate(sentence_gen(files)):
    if num % 10000 == 0:
        print(num)
    items = line.split("\t")
    items = items[1:]
    if len(items) < 3:
        continue
    for nu, item in enumerate(combinations(items, 3)):
        if nu > 3:
            break
        item = list(item)
        random.shuffle(item)
        if len(item[0].split()) < 2 or len(item[1].split()) < 2 or len(item[2].split()) < 2:
            continue
        data = item[0] + '\t' + item[1] + '\t' + item[2]
        i = 0
        while i < neg_number:
            neg_query = query_set.get_n_items()
            if item[0] not in neg_query and neg_query not in item[0]:
                data = data + '\t' + neg_query
                i += 1
        try:
            fo.write(data + '\n')
        except:
            print(data)
fo.close()



