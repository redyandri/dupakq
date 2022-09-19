import PyPDF2
import re
from tqdm import tqdm
from fuzzywuzzy import process as fuzz
import pandas as pd
import numpy as np
import regex as rex
import queue

pdf_f='data/juknis_prakom_2021.pdf'
reader = PyPDF2.PdfFileReader('data/juknis_prakom_2021.pdf')


def extract_activity_code(txt):
    act,code, label="","",""
    acts,codes, labels=[],[],[]
    try:
        txt=" ".join(re.findall("[\w\d.,\s]+",txt))
        acts=re.findall("[ivx]{1,3}\.[a-z]+\.*[a-z0-9]*\.*[a-z0-9]\.*[a-z0-9\s]+\n",str(txt).lower().strip())
        if acts:
            act=acts[0]
            codes=re.findall("[ivx]{1,3}\.[a-z]+\.*[a-z0-9]*\.*[a-z0-9]*\.*",str(act).strip())
            if codes:
                code=codes[0]
                if code[-1]==".":
                    code="".join(code[:-1])
                labels=re.findall("[ivx]{1,3}\.[a-z]+\.*[a-z0-9]*\.*[a-z0-9]*\.*(.*?)\n[a-z0-9\s]*",str(act).strip())
                if labels:
                    label=labels[0]
    except Exception as ex:
        print(str(ex))

    return act,code,label
act,code,label=extract_activity_code(reader.getPage(307).extractText())

df_dupak_all=pd.read_csv("data/dupak_all_with_blocks.csv", sep=";")

def get_activity_code_lower(txt):
    txt= txt.split("_")[-1].lower()
    if txt[-1]==".":
        txt="".join(txt[:-1])
    return txt
# get_activity_code_lower("Mahir_I.A.1.")
tqdm.pandas()
df_dupak_all["activity_code_only"]=df_dupak_all.activity_code.progress_apply(get_activity_code_lower)

def get_activity_last_part(txt):
    if txt[-1]==".":
        txt="".join(txt[:-1])
    return txt.split(".")[-1].lower()
# get_activity_last_part("Mahir_I.A.1.")
tqdm.pandas()
df_dupak_all["activity_last_part"]=df_dupak_all.activities.progress_apply(get_activity_last_part)

duplic_code=df_dupak_all[df_dupak_all.activity_code_only.duplicated()]["activity_code_only"].to_list()
dict_redundant_codes={k:[] for k in duplic_code}
for idx,row in df_dupak_all[df_dupak_all.activity_code_only.isin(duplic_code)][["activity_code","activity_code_only","activity_last_part"]].iterrows():
    dict_redundant_codes[row.activity_code_only].append(row.activity_code+"#"+row.activity_last_part)

def extract_activity_code2(txt):
    act,code, label="","",""
    acts,codes, labels=[],[],[]
    try:
        txt=" ".join(re.findall("[\w\d.,\s]+",txt))
        acts=re.findall("[ivx]{1,3}\.[a-z]+\.*[a-z0-9]*\.*[a-z0-9]\.*[a-z0-9\s]+\n",str(txt).lower().strip())
        if acts:
            act=acts[0]
            codes=re.findall("[ivx]{1,3}\.[a-z]+\.*[a-z0-9]*\.*[a-z0-9]*\.*",str(act).strip())
            if codes:
                code=codes[0].strip()
                if code[-1]==".":
                    code="".join(code[:-1]).strip()
                after_line_break="\n"+"\n".join(act.split("\n")[1:])
                labels=re.findall("(?<="+code+")(.*?)(?="+after_line_break+")",str(act))

                if labels:
                    label=labels[0].strip()

    except Exception as ex:
        print(str(ex))
    act=act.strip()
    code=code.strip()
    label=label.strip()
    return act,code,label
act,code,label=extract_activity_code2(reader.getPage(26).extractText())


duplicated_codes=[x for x in dict_redundant_codes.keys()]
pbar=tqdm(range(reader.numPages))
evident_prefix="mencakup namun tidak terbatas "
evident_suffix="contoh"
tmp_txt=""
regx="(?<="+evident_prefix+")[\s\S]+?(?="+evident_suffix+")"
regx2=evident_prefix+"\K[\s\S]+"
code_queue=queue.Queue()
for page in pbar:
    pbar.set_description("page {}".format(page))
    act=[]
    code=[]
    label=[]
    chosen_score,chosen_code,chosen_master=0,"",""
    try:
        txt=reader.getPage(page).extractText()
        act,code,label=extract_activity_code2(reader.getPage(page).extractText())
        if code:
            if code in duplicated_codes:
                masters=dict_redundant_codes[code]
                res=fuzz.extract(label,masters)
                scores=[score for pred,score in res]
                preds=[pred for pred,score in res]
                chosen_score=scores[0]
                chosen_master=preds[0]
                arr=chosen_master.split("#")
                code=arr[0]
                label=arr[1]
            code_queue.put([code,label])
        txt=txt.lower().replace("\n","")
        tmp_txt+=txt
        evidents=rex.findall(regx,tmp_txt)
        if evidents:
            evidents2=rex.findall(regx2,evidents[0])
            if evidents2:
                result=evidents2[0].replace("\n","").replace(";","")
                q=code_queue.get()
                kode=q[0]
                judul=q[1]
#                 df_dupak_all.loc[df_dupak_all.activity_code==kode,"evidents"]=result
                print("page {} {}:{} SCORE:{}, EVIDENT:{}".format(page,\
                            kode,judul,chosen_score,result))
                tmp_txt=""
        else:
            tmp_txt+="\n"+txt

    except Exception as ex:
        print(str(ex))

