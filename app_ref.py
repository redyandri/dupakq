from flask import Flask
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from flask import request
import flask
from victorinox import victorinox
import os
import logging
from PIL import Image
import cv2
from src import facenet
import tensorflow as tf
import jsonify
import sys

app = Flask(__name__,static_url_path='')
api = Api(app)
tool=victorinox()
facenet_model=r"C:\Users\redy.andriyansah\Documents\project\facenet\weight\20180402-114759.pb"#r"/home/andri/Documents/project/facenet/src/haarcascade_frontalface_alt2.xml"
classifier_model=r"C:\Users\redy.andriyansah\Documents\project\facenet\weight\pusintek200.pkl"#r"/home/andri/Documents/project/facenet/received"
receiver = r"C:\Users\redy.andriyansah\Documents\project\facenet\received"#r"/home/andri/Documents/project/facenet/weight/20180402-114759.pb"
haarcascade = r"C:\Users\redy.andriyansah\Documents\project\facenet\src\haarcascade_frontalface_alt2.xml"#r"/home/andri/Documents/project/facenet/weight/pusintek200.pkl"
seed=666
test_img_num=1
with open(facenet_model, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name='')
graph.finalize()
sess_config = tf.ConfigProto(
    log_device_placement=False,
    allow_soft_placement = True,
    # gpu_options = tf.GPUOptions(
    #     per_process_gpu_memory_fraction=1
    # )
)
sess = tf.Session(graph=graph,
                  config=sess_config)
np.random.seed(seed=seed)

# Load the model
print('Loading feature extraction model')
facenet.load_model(facenet_model)
with open(classifier_model, 'rb') as infile:
    (model, class_names) = pickle.load(infile)
print('Loaded classifier model from file "%s"' % classifier_model)

# Get input and output tensors
images_placeholder = graph.get_tensor_by_name("input:0")#tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = graph.get_tensor_by_name("embeddings:0")#tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = graph.get_tensor_by_name("phase_train:0") # tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]
emb_array = np.zeros((test_img_num, embedding_size))

@app.route('/photo', methods = ['POST'])
def photo():
    res={}
    nip="123"
    try:
        # file = Image.open(request.files['file'])
        # opencvImage = cv2.cvtColor(np.array(file), cv2.COLOR_RGB2BGR)
        # faces = tool.capture_face_from_matrice_by_haarcascade(haarcascade=haarcascade,
        #                                                       img_matrice=opencvImage,
        #                                                       rezisedim=(160, 160))
        imagefile=request.files.get('file')
        path=os.path.join(receiver,tool.get_id()+".jpg")
        imagefile.save(path)
        paths=tool.capture_face_from_image_by_haarcascade(haarcascade=haarcascade,
                                                          img_path=path,
                                                          dst_folder=receiver,
                                                          rezisedim=(160, 160))
        pred = []
        for face in paths:
            img = facenet.load_data([face], False, False, 160)

            # dim = img.shape
            # img = np.reshape(img, [1, dim[0], dim[1], dim[2]])
            print("IMAGE SHAPE:%s" % str(np.shape(img)))
            feed_dict = {images_placeholder: img, phase_train_placeholder: False}
            # sess = tf.Session(graph=graph,
            #                   config=sess_config)
            emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
            # sess.close()
            print('Testing classifier')
            predictions = model.predict_proba(emb_array)
            best_class_indices = np.argmax(predictions, axis=1)
            # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

            predicted = class_names[best_class_indices[0]]
            pred.append(predicted)


        app.logger.info(str(pred))
        res={"resp":pred}
    except Exception as err:
        # if sess:
        # sess.close()
        res={"error":str(err)}
    finally:
        # if sess:
        # sess.close()
        os.remove(path)
        for fff in paths:
            os.remove(fff)
    return res

@app.route('/face', methods = ['POST'])
def face():
    res={}
    nip="123"
    try:
        # file = Image.open(request.files['file'])
        # opencvImage = cv2.cvtColor(np.array(file), cv2.COLOR_RGB2BGR)
        # faces = tool.capture_face_from_matrice_by_haarcascade(haarcascade=haarcascade,
        #                                                       img_matrice=opencvImage,
        #                                                       rezisedim=(160, 160))
        imagefile=request.files.get('file')
        path=os.path.join(receiver,tool.get_id()+".jpg")
        imagefile.save(path)
        im=Image.open(path)
        im=im.resize((160,160),Image.NEAREST)
        im=im.convert("RGB")
        im.save(path)
        img = facenet.load_data([path], False, False, 160)

        # dim = img.shape
        # img = np.reshape(img, [1, dim[0], dim[1], dim[2]])
        print("IMAGE SHAPE:%s" % str(np.shape(img)))
        feed_dict = {images_placeholder: img, phase_train_placeholder: False}
        # sess = tf.Session(graph=graph,
        #                   config=sess_config)
        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
        # sess.close()
        print('Testing classifier')
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        pred = class_names[best_class_indices[0]]


        app.logger.info(str(pred))
        res={"resp":pred}
    except Exception as err:
        # if sess:
        # sess.close()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        res={"error":str(exc_type)+"\n"+str( fname)+"\n"+ str(exc_tb.tb_lineno)+"\n"+str(err)}
    finally:
        a=1
        # if sess:
        # sess.close()
        # os.remove(path)

    return res


@app.route('/')
def root():
    return app.send_static_file('index.html')
# model = NLPModel()
#
# clf_path = 'lib/models/SentimentClassifier.pkl'
# with open(clf_path, 'rb') as f:
#     model.clf = pickle.load(f)
#
# vec_path = 'lib/models/TFIDFVectorizer.pkl'
# with open(vec_path, 'rb') as f:
#     model.vectorizer = pickle.load(f)

# argument parsing
# parser = reqparse.RequestParser()
# parser.add_argument('query')


# class PredictSentiment(Resource):
#     def get(self):
#         # use parser and find the user's query
#         args = parser.parse_args()
#         user_query = args['query']
#
#         # vectorize the user's query and make a prediction
#         uq_vectorized = model.vectorizer_transform(np.array([user_query]))
#         prediction = model.predict(uq_vectorized)
#         pred_proba = model.predict_proba(uq_vectorized)
#
#         # Output either 'Negative' or 'Positive' along with the score
#         if prediction == 0:
#             pred_text = 'Negative'
#         else:
#             pred_text = 'Positive'
#
#         # round the predict proba value and set to new variable
#         confidence = round(pred_proba[0], 3)
#
#         # create JSON object
#         output = {'prediction': pred_text, 'confidence': confidence}
#
#         return output


# Setup the Api resource routing here
# Route the URL to the resource
# api.add_resource(PredictSentiment, '/')

@app.route('/face_saver', methods = ['POST'])
def face():
    res={}
    nip="123"
    try:
        # file = Image.open(request.files['file'])
        # opencvImage = cv2.cvtColor(np.array(file), cv2.COLOR_RGB2BGR)
        # faces = tool.capture_face_from_matrice_by_haarcascade(haarcascade=haarcascade,
        #                                                       img_matrice=opencvImage,
        #                                                       rezisedim=(160, 160))
        imagefile=request.files.get('file')
        path=os.path.join(receiver,tool.get_id()+".jpg")
        imagefile.save(path)
        im=Image.open(path)
        im=im.resize((160,160),Image.NEAREST)
        im=im.convert("RGB")
        im.save(path)
        img = facenet.load_data([path], False, False, 160)

        # dim = img.shape
        # img = np.reshape(img, [1, dim[0], dim[1], dim[2]])
        print("IMAGE SHAPE:%s" % str(np.shape(img)))
        feed_dict = {images_placeholder: img, phase_train_placeholder: False}
        # sess = tf.Session(graph=graph,
        #                   config=sess_config)
        emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
        # sess.close()
        print('Testing classifier')
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        # best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

        pred = class_names[best_class_indices[0]]


        app.logger.info(str(pred))
        res={"resp":pred}
    except Exception as err:
        # if sess:
        # sess.close()
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        res={"error":str(exc_type)+"\n"+str( fname)+"\n"+ str(exc_tb.tb_lineno)+"\n"+str(err)}
    finally:
        a=1
        # if sess:
        # sess.close()
        # os.remove(path)

    return res

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=53129,debug=True)
