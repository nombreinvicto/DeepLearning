#%%
import os

with open("collet_wrod2vec_test.txt", mode='r') as file:
    f = file.read()
#%%

lines = f.split(".")
print(lines)
#%%

len(lines)
#%%
lines_list = lines
#%%
lines_list[0].lower()
#%%
lower_case_list = [line.lower() for line in lines_list]
#%%
remove_special_chars = [lines.replace("\n", "") for lines in lower_case_list]
#%%
from string import punctuation
out_list = []

for line in remove_special_chars:
    l = line
    for char in punctuation:
        l = l.replace(char, "")
    out_list.append(l)
#%%
out_list
#%%
len(out_list)
#%%
with open("collet_wrod2vec_test_refined.txt", mode='a') as fileOut:
    for line in out_list:
        fileOut.write(line)
        fileOut.write("\n")

#%%
final_out = [line_list_.split(" ") for line_list_ in out_list]

 #%%
from gensim.models import Word2Vec

model = Word2Vec(sentences=final_out, min_count=1)

#%%
words = list(model.wv.vocab)
words
#%%
with open("collet_wrod2vec_test_refined.txt", mode='a') as fileOut:
    for word in words:
        fileOut.write(word)
        fileOut.write("\n")