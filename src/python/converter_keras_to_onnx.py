import onnx
from keras.models import load_model
import onnxmltools
import winmltools
import keras2onnx


model = load_model('c:\fer2013_mini_XCEPTION.107-0.66.hdf5')

onnx_model = onnxmltools.convert_keras(model, target_opset=7)

onnx.save(onnx_model, 'modelo_analise_sentimento.onnx')