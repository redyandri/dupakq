from fastapi import FastAPI, File
from starlette.requests import Request
import pickle
import numpy as np
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory,StopWordRemover,ArrayDictionary
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import random
from uuid import uuid4
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from fastapi import Body
import uvicorn
import datetime
import io
import json
from typing import List
from typing import Dict
from pydantic import BaseModel
import queue
from docxtpl import DocxTemplate
from io import StringIO
from starlette.responses import StreamingResponse
from uuid import uuid4
# import redis
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime


tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
factory = StemmerFactory()
stemmer = factory.create_stemmer()
# default_stopwords = StopWordRemoverFactory().get_stop_words()
# additional_stopwords=["(",")","senin","selasa","rabu","kamis","jumat","sabtu","minggu"]
# dictionary=ArrayDictionary(default_stopwords+additional_stopwords)
# id_stopword = StopWordRemover(dictionary)

en_stemmer = PorterStemmer()
en_stopwords = nltk.corpus.stopwords.words('english')

df_id_stopword=pd.read_csv("data/stopwordbahasa.csv",header=None)
id_stopword=df_id_stopword[0].to_list()

def tokenize_clean(text):
    if(text):
        tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word
            in nltk.word_tokenize(sent)]
        #clean token from numeric and other character like puntuation
        filtered_tokens = []
        for token in tokens:
            txt=re.findall('[a-zA-Z]{3,}', token)
            if txt:
                filtered_tokens.append(txt[0])
        return filtered_tokens

def remove_stopwords(tokenized_text):
    if(tokenized_text):
        cleaned_token = []
        for token in tokenized_text:
            if token not in id_stopword:
                cleaned_token.append(token)

        return cleaned_token

def stem_text(tokenized_text):
    if (tokenized_text):
        stems = []
        for token in tokenized_text:
            stems.append(stemmer.stem(token))

        return stems

def remove_en_stopwords(text):
    if text:
        return [token for token in text if token not in en_stopwords]

def stem_en_text(text):
    if text:
        return [en_stemmer.stem(word) for word in text]

def revome_slash_n(text):
    if text:
        return [str(txt).replace("\n"," ") for txt in text]

def lower_text(text):
    if text:
        return [str(txt).lower() for txt in text]

def make_sentence(arr):
    if arr:
        return " ".join(arr)

def text_preprocessing(text):
    if text:
        prep01 = tokenize_clean(text)
        prep02 = remove_stopwords(prep01)
        prep03 = stem_text(prep02)
        prep04 = remove_en_stopwords(prep03)
        prep05 = stem_en_text(prep04)
        prep06 = revome_slash_n(prep05)
        prep07 = lower_text(prep06)
        prep08 = make_sentence(prep07)
        return prep08

tfidf_vectorizer=TfidfVectorizer(use_idf=True,ngram_range=(1,2))
tfidf_model_fp="model/tfdif_vectorizer.pkl"
with open(tfidf_model_fp, "rb") as fi:
    tfidf_vectorizer=pickle.load(fi)
    print("tfidf loaded from ", tfidf_model_fp)

tfidf_model_fp_enriched="model/tfdif_vectorizer_enriched.pkl"
with open(tfidf_model_fp_enriched, "rb") as fi:
    tfidf_vectorizer_enriched=pickle.load(fi)
    print("tfidf_enriched loaded from ", tfidf_model_fp_enriched)

tfidf_block_model_fp="model/tfidf_block.pkl"
with open(tfidf_block_model_fp, "rb") as fi:
    tfidf_vectorizer_block=pickle.load(fi)
    print("tfidf_block loaded from ", tfidf_block_model_fp)

knn_model_path="model/knn.pkl"
knn_index_path="model/knn_idx.pkl"
with open(knn_model_path,"rb") as fi:
  knn=pickle.load(fi)
  print("KNN loaded from ",knn_model_path)
with open(knn_index_path,"rb") as fi:
  knn_index=pickle.load(fi)
  print("Index KNN loaded from ",knn_index_path)

knn_model_path_enriched="model/knn_enriched.pkl"
knn_index_path_enriched="model/knn_idx_enriched.pkl"
with open(knn_model_path_enriched,"rb") as fi:
  knn_enriched=pickle.load(fi)
  print("KNN loaded from ",knn_model_path_enriched)
with open(knn_index_path_enriched,"rb") as fi:
  knn_index_enriched=pickle.load(fi)
  print("Index KNN loaded from ",knn_index_path_enriched)

knn_block_index_path="model/knn_block_idx.pkl"
with open(knn_block_index_path,"rb") as fi:
  knn_block_index=pickle.load(fi)
  print("KNN index loaded from ",knn_block_index_path)
knn_block_path="model/knn_block.pkl"
with open(knn_block_path,"rb") as fi:
  knn_block=pickle.load(fi)
  print("KNN loaded from ",knn_block_path)

def vectorize_tfidf(txt):
    densematrix=tfidf_vectorizer.transform([txt])
    skillvecs=densematrix.toarray().tolist()
    vector=np.array(skillvecs[0]).astype('float32').tolist()
    return vector

def vectorize_tfidf_enriched(txt):
    densematrix=tfidf_vectorizer_enriched.transform([txt])
    skillvecs=densematrix.toarray().tolist()
    vector=np.array(skillvecs[0]).astype('float32').tolist()
    return vector

def vectorize_tfidf_block(txt):
    densematrix=tfidf_vectorizer_block.transform([txt])
    skillvecs=densematrix.toarray().tolist()
    vector=np.array(skillvecs[0]).astype('float32').tolist()
    return vector

df_dupak_all=pd.read_csv("data/dupak_all.csv", sep=";")

kamus = {}
with open("data/dict.json") as fi:
    kamus = json.load(fi)

kamus_normalized = {}
with open("data/kamus_normalized.json") as fi:
    kamus_normalized = json.load(fi)

def enrich_activity(txt):
    txt = str(txt).lower()
    arr = re.findall("\w+", txt)
    final_result = " ".join(arr)
    template = final_result
    for token in arr:
        syns = []
        try:
            syns = kamus[token]["sinonim"]
        except:
            try:
                syns = kamus_normalized[token]["sinonim"]
            except:
                pass
        for syn in syns:
            if syn is not None:
                a = str(template).replace(token, syn)
                final_result += ". " + a
    return final_result




class Response_Item(BaseModel):
    code: str
    activity: str
    level: str
    credit: float

class Responses(BaseModel):
    val: Dict[str, Response_Item]

docx_dict={}
template = "data/TEMPLATE2.docx"
session_id=""

# red = redis.Redis(
#     host='127.0.0.1',
#     port=6379,
#     password=''
# )

# def get_docx_link(idx):
#     tpl = DocxTemplate(template)
#     context = {}
#     context["nip_prakom"] = "198401112009011004"
#     context["nip_atasan"] = "198401112009011005"
#     context["kode_kegiatan"] = "III.B.1"
#     context["bukti_kegiatan"] = "1) a, \n\n\n2) b\n\n\n, 3) c\n\n\n"
#     context["judul_kegiatan"] = "pembuatan model AI"
#     context["lokasi"] = "karlsruhe"
#     context["query_kegiatan"] = "buat model machine learning"
#     context["keterangan_kegiatan"] = "-"
#     context["nama_atasan"] = "ismail fahmi"
#     context["nama_prakom"] = "redy andri"
#     context["pangkat"] = "Penaa Tk. I"
#     context["jenjang_prakom"] = "ahli muda"
#     context["tanggal"] = "23-agustus-2022"
#     context["angka_kredit"] = "1,25"
#     context["golongan"] = "IIIB"
#     tpl.render(context)
#     # Create in-memory buffer
#     file_stream = io.BytesIO()
#     # Save the .docx to the buffer
#     tpl.save(file_stream)
#     # Reset the buffer's file-pointer to the beginning of the file
#     file_stream.seek(0)
#     headers = {
#         'Content-Disposition': 'attachment; filename="filename.docx"'
#     }
#     return StreamingResponse(file_stream, headers=headers)

def ge_activity_code(txt):
    arr=re.findall("[a-z0-9]{1,2}\.[a-z0-9]{1,2}\.*[a-z0-9]*\.*[a-z0-9]*", str(txt).lower())
    if arr:
        return arr[0]
    else:
        return []

app = FastAPI()
origins = ["*"]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class Query(BaseModel):
    q: str

@app.post("/test/")
async def test(request: Request):
    return {"response":await request.headers.get('Content-Type')}

@app.post("/search/")
# async def search(q: str=Body(embed=True)):
async def search(request: Request):
    result = {}
    # form = await request.form()
    form=await request.json()
    q = str(form["q"])
    # q = str(query["q"])
    if q:
        merge_result = {"vanilla": {}, "enriched": {}}
        skipped_idx_queue = queue.Queue()
        skipped_distance_queue= queue.Queue()
        q_cleansed = text_preprocessing(q)
        q_vector = vectorize_tfidf(q_cleansed)
        (v_distances, v_indices) = knn.kneighbors([q_vector], n_neighbors=5)
        v_indices = v_indices.tolist()
        res = [knn_index[x] for x in v_indices[0]]
        for i, x in enumerate(res):
            merge_result["vanilla"][i] = {"index": v_indices[0][i], "distance": v_distances[0][i],
                                          "activity": x, "ak": df_dupak_all.loc[v_indices[0][i], ["ak"]][0]}

        # q2=enrich_activity(q)
        # q_cleansed=text_preprocessing(q2)
        q_vector = vectorize_tfidf_enriched(q_cleansed)
        (e_distances, e_indices) = knn_enriched.kneighbors([q_vector], n_neighbors=5)
        e_indices = e_indices.tolist()
        res = [knn_index_enriched[x] for x in e_indices[0]]
        for i, x in enumerate(res):
            merge_result["enriched"][i] = {"index": e_indices[0][i], "distance": e_distances[0][i], "activity": x,
                                           "ak": df_dupak_all.loc[e_indices[0][i], ["ak"]][0]}
        elapsed_idx = []
        skipped_idx = []

        vanilla_index = merge_result["vanilla"]
        total_idx = v_indices[0] + e_indices[0]
        # intersected_index=list(set(v_indices[0]).intersection(e_indices[0]))
        elapsed_idx = []
        skipped_idx = []
        result = {}

        # counter=1
        result_num = 5
        for i in range(5):
            tmp = {}
            d = 99
            if merge_result["vanilla"][i]["distance"] < merge_result["enriched"][i]["distance"]:
                idx = merge_result["vanilla"][i]["index"]
                if idx not in elapsed_idx:
                    if merge_result["vanilla"][i]["index"] != merge_result["enriched"][i]["index"]:
                        skipped_idx = merge_result["enriched"][i]["index"]
                        skipped_d = merge_result["enriched"][i]["distance"]

                    else:
                        skipped_idx = -1
                    tmp["distance"] = merge_result["vanilla"][i]["distance"]
                else:
                    if not skipped_idx_queue.empty():
                        idx = skipped_idx_queue.get()
                        d = skipped_distance_queue.get()
                        tmp["distance"] = d
                tmp["model"] = "vanilla"
            else:
                idx = merge_result["enriched"][i]["index"]
                if idx not in elapsed_idx:
                    if merge_result["vanilla"][i]["index"] != merge_result["enriched"][i]["index"]:
                        skipped_idx = merge_result["vanilla"][i]["index"]
                        skipped_d = merge_result["vanilla"][i]["distance"]

                    else:
                        skipped_idx = -1
                    tmp["distance"] = merge_result["enriched"][i]["distance"]
                else:
                    if not skipped_idx_queue.empty():
                        idx = skipped_idx_queue.get()
                        d = skipped_distance_queue.get()
                        tmp["distance"] = d
                tmp["model"] = "enriched"

            tmp["index"] = idx
            tmp["code"] = df_dupak_all.loc[idx, "activity_code"]
            tmp["activity"] = df_dupak_all.loc[idx, "activities"]
            tmp["level"] = df_dupak_all.loc[idx, "jenjang"]
            tmp["credit"] = df_dupak_all.loc[idx, "ak"]
            # tmp["session"]=get_docx_link(idx)
            result[i] = tmp
            elapsed_idx.append(idx)
            if skipped_idx != -1:
                skipped_idx_queue.put(skipped_idx)
                skipped_distance_queue.put(skipped_d)
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/search2/")
async def search2(request: Request):
    result = {}
    # form = await request.form()
    form=await request.json()
    q = str(form["q"])
    # q = str(query["q"])
    if q:
        result={}
        merge_result = {"vanilla": {}, "enriched": {},"block":{}}
        skipped_idx_queue = queue.Queue()
        skipped_distance_queue= queue.Queue()
        activitycode=ge_activity_code(q)
        if activitycode:
            actvity_code_only=activitycode
            dftmp = df_dupak_all.query("actvity_code_only==@actvity_code_only")
            for a,i in enumerate(dftmp.index):
                tmp = {}
                try:
                    tmp["distance"] = 0.0
                    tmp["model"] = "code_search"
                    tmp["index"] = str(i)+"_"+df_dupak_all.loc[i,"activity_code"]
                    tmp["code"] = df_dupak_all.loc[i,"actvity_code_only"]
                    tmp["activity"] = df_dupak_all.loc[i, "activity_last_part"]
                    tmp["level"] = df_dupak_all.loc[i, "jenjang"]
                    tmp["credit"] = float(df_dupak_all.loc[i, "ak"])
                except:
                    tmp["distance"] = 0.0
                    tmp["model"] = "-"
                    tmp["index"] = "-"
                    tmp["code"] = "-"
                    tmp["activity"] = "-"
                    tmp["level"] = "-"
                    tmp["credit"] = 0.0
                result[str(a)]=tmp
        else:
            q_cleansed = text_preprocessing(q)
            q_vector = vectorize_tfidf(q_cleansed)
            (v_distances, v_indices) = knn.kneighbors([q_vector], n_neighbors=5)
            v_indices = v_indices.tolist()
            res = [knn_index[x] for x in v_indices[0]]
            for i, x in enumerate(res):
                merge_result["vanilla"][i] = {"index": v_indices[0][i], "distance": v_distances[0][i],
                                              "activity": x, "ak": df_dupak_all.loc[v_indices[0][i], ["ak"]][0]}

            # q2=enrich_activity(q)
            # q_cleansed=text_preprocessing(q2)
            q_vector = vectorize_tfidf_enriched(q_cleansed)
            (e_distances, e_indices) = knn_enriched.kneighbors([q_vector], n_neighbors=5)
            e_indices = e_indices.tolist()
            res = [knn_index_enriched[x] for x in e_indices[0]]
            for i, x in enumerate(res):
                merge_result["enriched"][i] = {"index": e_indices[0][i], "distance": e_distances[0][i], "activity": x,
                                               "ak": df_dupak_all.loc[e_indices[0][i], ["ak"]][0]}

            # q_vector3 = vectorize_tfidf_block(q_cleansed)
            # (e_distances3, e_indices3) = knn_block.kneighbors([q_vector3], n_neighbors=5)
            # e_indices3 = e_indices3.tolist()
            # res = [knn_block_index[x] for x in e_indices3[0]]
            # for i, x in enumerate(res):
            #     merge_result["block"][i] = {"index": e_indices3[0][i], "distance": e_distances3[0][i], "activity": x,
            #                                    "ak": df_dupak_all.loc[e_indices3[0][i], ["ak"]][0]}

            vanilla_index = merge_result["vanilla"]
            total_idx = v_indices[0] + e_indices[0]
            intersected_index=list(set(v_indices[0]).intersection(e_indices[0]))
            elapsed_idx = []
            skipped_idx = []
            result = {}

            # counter=1
            result_num = 5
            for i in range(5):
                tmp = {}
                d = 99
                if merge_result["vanilla"][i]["distance"] < merge_result["enriched"][i]["distance"]:
                    idx = merge_result["vanilla"][i]["index"]
                    if idx not in elapsed_idx:
                        if merge_result["vanilla"][i]["index"] != merge_result["enriched"][i]["index"]:
                            skipped_idx = merge_result["enriched"][i]["index"]
                            skipped_d = merge_result["enriched"][i]["distance"]

                        else:
                            skipped_idx = -1
                        tmp["distance"] = merge_result["vanilla"][i]["distance"]
                    else:
                        if not skipped_idx_queue.empty():
                            idx = skipped_idx_queue.get()
                            d = skipped_distance_queue.get()
                            tmp["distance"] = d
                    tmp["model"] = "vanilla"
                else:
                    idx = merge_result["enriched"][i]["index"]
                    if idx not in elapsed_idx:
                        if merge_result["vanilla"][i]["index"] != merge_result["enriched"][i]["index"]:
                            skipped_idx = merge_result["vanilla"][i]["index"]
                            skipped_d = merge_result["vanilla"][i]["distance"]

                        else:
                            skipped_idx = -1
                        tmp["distance"] = merge_result["enriched"][i]["distance"]
                    else:
                        if not skipped_idx_queue.empty():
                            idx = skipped_idx_queue.get()
                            d = skipped_distance_queue.get()
                            tmp["distance"] = d
                    tmp["model"] = "enriched"

                c=df_dupak_all.loc[idx, "activity_code"]
                tmp["index"] = str(idx)+"_"+c
                tmp["code"] = c
                tmp["activity"] = df_dupak_all.loc[idx, "activity_last_part"]
                tmp["level"] = df_dupak_all.loc[idx, "jenjang"]
                tmp["credit"] = float(df_dupak_all.loc[idx, "ak"])
                # tmp["session"]=get_docx_link(idx)
                result[i] = tmp
                elapsed_idx.append(idx)
                if skipped_idx != -1:
                    skipped_idx_queue.put(skipped_idx)
                    skipped_distance_queue.put(skipped_d)

    arr=[]
    for i,res in result.items():
        arr.append(res)
    result2={"results":arr}
    json_compatible_item_data = jsonable_encoder(result2)
    # json_compatible_item_data = jsonable_encoder(final)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/download/", response_description='docx')
async def download(request: Request):
    result = {}
    # form = await request.form()
    form = await request.json()
    idx = str(form["idx"])
    act = str(form["act"])
    # q = str(query["q"])
    if idx:
            d = datetime.now().date()
            tpl = DocxTemplate(template)
            idx=int(idx)
            context = {}
            # context["nip_prakom"] = "198401112009011004"
            # context["nip_atasan"] = "198401112009011005"
            ac=df_dupak_all.at[idx,"activity_code"]
            arrs=ac.split("_")
            jenjang=arrs[0]
            cd=arrs[1]
            doc_title=cd+"_"+act
            evdnt=eval(df_dupak_all.at[idx,"evidents"])
            evdnt=[str(i)+") "+x for i,x in enumerate(evdnt) if x]
            # print("doc_title:{}".format(doc_title))
            context["kode_kegiatan"] = cd
            context["bukti_kegiatan"] = "\n\n\n".join(evdnt)
            context["judul_kegiatan"] = df_dupak_all.at[idx,"activity_last_part"]
            context["lokasi"] = "Jakarta"
            context["query_kegiatan"] = act
            context["keterangan_kegiatan"] = "-"
            context["nama_atasan"] = "@nama_atasan"
            context["nama_prakom"] = "@nama_prakom"
            context["pangkat"] = "@pangkat_prakom"
            context["jenjang_prakom"] = jenjang
            context["tanggal"] = str(d.day)+"-"+str(d.month)+"-"+str(d.year)
            context["angka_kredit"] = df_dupak_all.at[idx,"ak"]
            context["golongan"] = "@golongan_prakom"
            tpl.render(context)
            # tpl.save("doctpl.docx")

            # Create in-memory buffer
            file_stream = io.BytesIO()
            # Save the .docx to the buffer
            tpl.save(file_stream)
            # Reset the buffer's file-pointer to the beginning of the file
            file_stream.seek(0)
            doc_title=doc_title.replace(" ","_")+".docx"
            headers = {
                'Content-Disposition': "'attachment; filename="+doc_title
            }
            return StreamingResponse(file_stream, headers=headers)


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)#, reload=True)
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)