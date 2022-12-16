# import redis


# r = redis.Redis(
#     host='127.0.0.1',
#     port=6379,
#     password=''
# )
# r.mset({"Croatia": "Zagreb", "Bahamas": "Nassau"})
# print(r.get("Bahamas"))
# print(r.exists("Bahamas2"))
# import pickle
# knn_index_path="model/knn_idx.pkl"
# with open(knn_index_path,"rb") as fi:
#   knn_index=pickle.load(fi)
#   print("Index KNN loaded from ",knn_index_path)
# for k,v in knn_index.items():
#     print('{}:{}'.format(k,v))

# import random
# print("#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)]))

import re

txt='kebo'
arr = re.findall("[a-z0-9]{1,2}\.[a-z0-9]{1,2}\.*[a-z0-9]*\.*[a-z0-9]*", str(txt).lower())
print(arr)  