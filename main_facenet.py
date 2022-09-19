from fastapi import FastAPI, File
from starlette.requests import Request
import io
import onnx
import onnxruntime as ort
from onnx_tf.backend import prepare
import tensorflow as tf
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import datetime
import src.facenet as facenet
import uvicorn
import cv2
from scipy import spatial
import pymongo
import numpy as np
import json
from confluent_kafka import Producer, Consumer, KafkaError

p = Producer({'bootstrap.servers': '10.235.4.134:9092,10.235.4.137:9092,10.235.4.146:9092,10.235.4.147:9092,10.235.4.148:9092'})

connection = pymongo.MongoClient('mongodb://app_it_facerecog:f4c3IT2021##@dc01-mongo.kemenkeu.go.id:27017/',authSource='it_facerecog')
db_facenet = connection.it_facerecog
col_average_vector=db_facenet.col_average_vector

detector_model=r"../../corpus/facenet/weight/ultra_light_320.onnx"
facenet_model =r"../../corpus/facenet/weight/20180402-114759.pb"
model_folder=r"../../corpus/facenet/models/average/ue2_staffs_and_leaders"
nip_id_name_org_pkl=r"../../corpus/facenet/models/nip_id_name_org.pkl"
id_nip_name_org_pkl=r"../../corpus/facenet/models/id_nip_name_org.pkl"
dev_path = r"../../corpus/facenet/dev"
error_log=r"../../corpus/facenet/errors/15_errors.txt"

onnx_model = onnx.load(detector_model)
predictor = prepare(onnx_model)
ort_session = ort.InferenceSession(detector_model)
input_name = ort_session.get_inputs()[0].name
onnx_dim=(320,240)

def detect_face_using_onnx(img_bgr):
    result=(None,[0,0,0,0])
    img_onnx_input = cv2.resize(img_bgr, onnx_dim)
    img_onnx_input_rgb = cv2.cvtColor(img_onnx_input, cv2.COLOR_BGR2RGB)
    img_mean = np.array([127, 127, 127])
    img = (img_onnx_input_rgb - img_mean) / 128
    img = np.transpose(img, [2, 0, 1])
    img = np.expand_dims(img, axis=0)
    img = img.astype(np.float32)
    confidences, boxes = ort_session.run(None, {input_name: img})
    w, h = onnx_dim[0], onnx_dim[1]
    boxes, labels, probs = predict(w, h, confidences, boxes, 0.7)
    for i in range(boxes.shape[0]):
        box = boxes[i, :]
        x1, y1, x2, y2 = box
        x1=x1 if x1>=0 else 0
        y1 = y1 if y1 >= 0 else 0
        x2 = x2 if x2 >= 0 else 0
        y2 = y2 if y2 >= 0 else 0
        break
    if boxes.shape[0] > 0:
        face_rgb = img_onnx_input_rgb[y1:y2, x1:x2]
        shape = np.shape(face_rgb)
        face_bgr = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2BGR)
        resized = cv2.resize(face_bgr, (shape[0] + 13, shape[1]), interpolation=cv2.INTER_AREA)  ####convert from landscape to portray face shape
        result = (resized, [x1, y1, x2, y2])
    return result

def area_of(left_top, right_bottom):
    hw = np.clip(right_bottom - left_top, 0.0, None)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    overlap_left_top = np.maximum(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = np.minimum(boxes0[..., 2:], boxes1[..., 2:])
    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    indexes = np.argsort(scores)
    indexes = indexes[-candidate_size:]
    while len(indexes) > 0:
        current = indexes[-1]
        picked.append(current)
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[:-1]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            np.expand_dims(current_box, axis=0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]

def predict(width, height, confidences, boxes, prob_threshold, iou_threshold=0.5, top_k=-1):
    boxes = boxes[0]
    confidences = confidences[0]
    picked_box_probs = []
    picked_labels = []
    for class_index in range(1, confidences.shape[1]):
        probs = confidences[:, class_index]
        mask = probs > prob_threshold
        probs = probs[mask]
        if probs.shape[0] == 0:
            continue
        subset_boxes = boxes[mask, :]
        box_probs = np.concatenate([subset_boxes, probs.reshape(-1, 1)], axis=1)
        box_probs = hard_nms(box_probs,
           iou_threshold=iou_threshold,
           top_k=top_k,
           )
        picked_box_probs.append(box_probs)
        picked_labels.extend([class_index] * box_probs.shape[0])
    if not picked_box_probs:
        return np.array([]), np.array([]), np.array([])
    picked_box_probs = np.concatenate(picked_box_probs)
    picked_box_probs[:, 0] *= width
    picked_box_probs[:, 1] *= height
    picked_box_probs[:, 2] *= width
    picked_box_probs[:, 3] *= height
    return picked_box_probs[:, :4].astype(np.int32), np.array(picked_labels), picked_box_probs[:, 4]

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

with open(facenet_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
graph.finalize()
sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement=True,
)
sess = tf.Session(graph=graph,
                  config=sess_config)
np.random.seed(seed=1)
# Load the model
print('Loading feature extraction model')
facenet.load_model(facenet_model)

test_img_num = 1
images_placeholder = graph.get_tensor_by_name("input:0")  # tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = graph.get_tensor_by_name("embeddings:0")  # tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")  # tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
emb_array_using_io = np.zeros((test_img_num, embedding_size)).astype("float32")
emb_array_using_cv2 = np.zeros((test_img_num, embedding_size)).astype("float32")
nearest_neighbor = 1
facenet_dim=(160, 160)
T=0.10   #0.65 #0.5 #0.76
arr=col_average_vector.find()
average_unmasked_model={}
for x in arr:
    average_unmasked_model[x['id_hris']]=x['average_vector']  #convert "[]" from mngodb to [] in python

class Response_Item(BaseModel):
    status: str
    timestamp: datetime.datetime
    prediction: str
    elapsed: float

class Registration_Response(BaseModel):
    status: str
    timestamp: datetime.datetime
    message: str

class Verify_Response_Item(BaseModel):
    status: str
    timestamp: datetime.datetime
    prediction: str
    elapsed: float
    score: float

app = FastAPI()
@app.post("/register/")
async def verify(request: Request,
                 face: bytes = File(...)):
    emb_array_using_cv2 = np.zeros((test_img_num, embedding_size)).astype("float32")
    is_verified = False
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not registered"
    }
    # result={}
    result = Registration_Response(**init_result)
    if request.method == "POST":
        try:
            t0 = datetime.datetime.now()
            form = await request.form()
            id=int(form["id"])
            if id in average_unmasked_model:
                result.status = "User already exist"
                result.message = ""
            # elif col_average_vector.find_one({"id_hris":id}):
            #     result.status = "User already exist"
            #     result.message = ""
            else:
                # absent_date=form["absent_date"]     #format '2018-06-29 08:15:27.243860'
                stream = io.BytesIO(face)
                data = np.fromstring(stream.getvalue(), dtype=np.uint8)
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                # (face_bgr, [x1, y1, x2, y2]) = detect_face_using_onnx(img_bgr)
                (face_bgr, _) = detect_face_using_onnx(img_bgr)
                if face_bgr is not None:
                    facenet_sized = cv2.resize(face_bgr, facenet_dim, interpolation=cv2.INTER_AREA)
                    facenet_sized_rgb = cv2.cvtColor(facenet_sized, cv2.COLOR_BGR2RGB)
                    facenet_sized_rgb_prewhitened = prewhiten(facenet_sized_rgb)
                    nrof_samples = 1
                    imgbatch = np.zeros((nrof_samples, facenet_dim[0], facenet_dim[1], 3))
                    imgbatch[0, :, :, :] = facenet_sized_rgb_prewhitened
                    feed_dict = {images_placeholder: imgbatch, phase_train_placeholder: False}
                    emb_array_using_cv2[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    emb_array_using_cv2=np.reshape(emb_array_using_cv2,-1)
                    vec=[float(x) for x in emb_array_using_cv2]
                    col_average_vector.insert_one({"id_hris":int(id),"average_vector":vec})  #insert to DB
                    average_unmasked_model[id] = vec                                         # insert to memory
                    p.poll(0)
                    data = json.dumps({"id": id, "vec": vec})
                    p.produce('facecreate', key="facecreate", value=data.encode('utf-8'), callback=delivery_report)
                    p.flush()
                    result.timestamp = datetime.datetime.now()#strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                    result.status = "ok"
                    result.message="User {} berhasil didaftarkan".format(str(id))
                else:
                    result.status = "Face NOT found"
                    result.message = "Face NOT found"

        except Exception as e:
            result.status = "Error"
            result.message="{}".format(str(e))
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/verify/")
async def verify(request: Request,
                 face: bytes = File(...)):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        'prediction': "-1",
        "elapsed": 0.0,
        "score":0.0
    }
    is_verified = False
    result = Verify_Response_Item(**init_result)
    if request.method == "POST":
        try:
            t0 = datetime.datetime.now()
            form = await request.form()
            id=int(form["id"])
            # absent_date=form["absent_date"]     #format '2018-06-29 08:15:27.243860'
            stream = io.BytesIO(face)
            data = np.fromstring(stream.getvalue(), dtype=np.uint8)
            img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            (face_bgr, _ ) = detect_face_using_onnx(img_bgr)
            avg_vec=[]
            if face_bgr is not None:
                facenet_sized = cv2.resize(face_bgr, facenet_dim, interpolation=cv2.INTER_AREA)
                facenet_sized_rgb = cv2.cvtColor(facenet_sized, cv2.COLOR_BGR2RGB)
                facenet_sized_rgb_prewhitened = prewhiten(facenet_sized_rgb)
                nrof_samples = 1
                imgbatch = np.zeros((nrof_samples, facenet_dim[0], facenet_dim[1], 3))
                imgbatch[0, :, :, :] = facenet_sized_rgb_prewhitened
                feed_dict = {images_placeholder: imgbatch, phase_train_placeholder: False}
                emb_array_using_cv2[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                try:
                    avg_vec = average_unmasked_model[id]
                except Exception as IDNotFoundException:
                    doc = col_average_vector.find_one({"id_hris": id})
                    if doc:
                        avg_vec=doc['average_vector']
                        average_unmasked_model[id]=avg_vec  ###update recen memory based on DB
                    else:
                        result.status = "Error"
                        # result.timestamp = datetime.datetime.strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                        elapsed = (datetime.datetime.now() - t0).total_seconds()
                        result.elapsed = elapsed
                        result.prediction = "User NOT found"
                sim_score = 1 - spatial.distance.cosine(avg_vec, emb_array_using_cv2)
                if sim_score > T:
                    is_verified = True
                else:
                    is_verified = False
                elapsed = (datetime.datetime.now() - t0).total_seconds()
                result.status = "OK"
                # result.timestamp = datetime.datetime.strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                result.elapsed = elapsed
                result.prediction = "verified" if is_verified else "not verified"
                result.score=sim_score
            else:
                result.status = "Error"
                # result.timestamp = datetime.datetime.strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                elapsed = (datetime.datetime.now() - t0).total_seconds()
                result.elapsed = elapsed
                result.prediction = "Face NOT found"
        except Exception as e:
            result.status = "error:%s"%str(e)
            result.timestamp = datetime.datetime.now()
            print("ERROR SERVER:%s" % str(e))

    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

class Response_Item2(BaseModel):
    status: str
    is_exist:bool

@app.post("/exist/")
async def exist(request: Request):
    init_result2 = {
        'status': 'Ok',
        'is_exist': False
    }
    is_exist = False
    result = Response_Item2(**init_result2)
    if request.method == "POST":
        try:
            form = await request.form()
            id=int(form["id"])
            try:
                avg_vec = average_unmasked_model[id]
                result.status = "Ok"
                result.is_exist = True
            except Exception as IDNotFoundException:
                doc = col_average_vector.find_one({"id_hris": id})
                if doc:
                    avg_vec=doc['average_vector']
                    average_unmasked_model[id]=avg_vec  ###update recen memory based on DB
                    result.status = "Ok"
                    result.is_exist = True
                else:
                    result.status = "Ok"
                    result.is_exist = False
        except Exception as e:
            result.status = "error:%s"%str(e)
            result.is_exist = False
            print("ERROR SERVER:%s" % str(e))

    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

def delivery_report(err, msg):
    """ Called once for each message produced to indicate delivery result.
        Triggered by poll() or flush(). """
    if err is not None:
        print('Message delivery failed: {}'.format(err))
    else:
        print('Message delivered to {} [{}]'.format(msg.topic(), msg.partition()))

@app.post("/update/")
async def update(request: Request,
                 face: bytes = File(...)):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not registered"
    }
    emb_array_using_cv2 = np.zeros((test_img_num, embedding_size)).astype("float32")
    is_verified = False
    result={}
    result = Registration_Response(**init_result)
    if request.method == "POST":
        try:
            t0 = datetime.datetime.now()
            form = await request.form()
            id=int(form["id"])
            if (id in average_unmasked_model) or (col_average_vector.find_one({"id_hris":id})):
                stream = io.BytesIO(face)
                data = np.fromstring(stream.getvalue(), dtype=np.uint8)
                img_bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
                (face_bgr, _) = detect_face_using_onnx(img_bgr)
                if face_bgr is not None:
                    facenet_sized = cv2.resize(face_bgr, facenet_dim, interpolation=cv2.INTER_AREA)
                    facenet_sized_rgb = cv2.cvtColor(facenet_sized, cv2.COLOR_BGR2RGB)
                    facenet_sized_rgb_prewhitened = prewhiten(facenet_sized_rgb)
                    nrof_samples = 1
                    imgbatch = np.zeros((nrof_samples, facenet_dim[0], facenet_dim[1], 3))
                    imgbatch[0, :, :, :] = facenet_sized_rgb_prewhitened
                    feed_dict = {images_placeholder: imgbatch, phase_train_placeholder: False}
                    emb_array_using_cv2[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    emb_array_using_cv2 = np.reshape(emb_array_using_cv2, -1)
                    vec = [float(x) for x in emb_array_using_cv2]
                    col_average_vector.update_one({"id_hris": int(id)},{'$set': {'average_vector': vec}}, upsert=False)
                    average_unmasked_model[id] = vec  # update to memory
                    p.poll(0)
                    data=json.dumps({"id":id,"vec":vec})
                    p.produce('faceupdate', key="faceupdate",value=data.encode('utf-8'), callback=delivery_report)
                    p.flush()
                    result.timestamp = datetime.datetime.now()  # strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                    result.status = "ok"
                    result.message = "Face vector {} berhasil diupdate".format(str(id))
                else:
                    result.status = "error"
                    result.message = "Face NOT found"
            else:
                result.status = "error"
                result.message = "User Not Found"


        except Exception as e:
            result.status = "error"
            result.message="{}".format(str(e))
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)


@app.post("/delete/")
async def delete(request: Request):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not registered"
    }
    result={}
    result = Registration_Response(**init_result)
    if request.method == "POST":
        try:
            t0 = datetime.datetime.now()
            form = await request.form()
            id=int(form["id"])
            if average_unmasked_model[id]:
                del average_unmasked_model[id]
                col_average_vector.delete_many({"id_hris": int(id)})
                p.poll(0)
                data=json.dumps({"id":id})
                p.produce('facedelete', key="facedelete",value=data.encode('utf-8'), callback=delivery_report)
                p.flush()
                result.timestamp = datetime.datetime.now()  # strptime(absent_date, '%Y-%m-%d %H:%M:%S.%f')
                result.status = "ok"
                result.message = "Face vector {} berhasil didelete".format(str(id))
            else:
                result.status = "error"
                result.message = "User Not Found"
        except Exception as e:
            result.status = "error"
            result.message="{}".format(str(e))
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

class Item(BaseModel):
    id: int
    vec: list

@app.post("/create_cache/")
async def update_cache(item: Item):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not updated"
    }
    result = Registration_Response(**init_result)
    try:
        id=int(item.id)
        vec=item.vec
        average_unmasked_model[id] = vec
        result.status="ok"
        result.message="cache facevector {} is created".format(id)
        result.timestamp = datetime.datetime.now()
    except Exception as e:
        result.status="error"
        result.message="Error: {}".format(str(e))
        result.timestamp = datetime.datetime.now()
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/update_cache/")
async def update_cache(item: Item):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not updated"
    }
    result = Registration_Response(**init_result)
    try:
        id=int(item.id)
        vec=item.vec
        average_unmasked_model[id] = vec
        result.status="ok"
        result.message="cache facevector {} is updated".format(id)
        result.timestamp = datetime.datetime.now()
    except Exception as e:
        result.status="error"
        result.message="Error: {}".format(str(e))
        result.timestamp = datetime.datetime.now()
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

@app.post("/delete_cache/")
async def update_cache(item: Item):
    init_result = {
        'status': 'error',
        'timestamp': datetime.datetime.now(),
        "message": "not updated"
    }
    result = Registration_Response(**init_result)
    try:
        id=int(item.id)
        if average_unmasked_model[id]:
            del average_unmasked_model[id]
            result.status="ok"
            result.message="cache facevector {} is deleted".format(id)
            result.timestamp = datetime.datetime.now()
        else:
            result.status = "ok"
            result.message = "user not found"
            result.timestamp = datetime.datetime.now()
    except Exception as e:
        result.status="error"
        result.message="Error: {}".format(str(e))
        result.timestamp = datetime.datetime.now()
    json_compatible_item_data = jsonable_encoder(result)
    return JSONResponse(content=json_compatible_item_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
