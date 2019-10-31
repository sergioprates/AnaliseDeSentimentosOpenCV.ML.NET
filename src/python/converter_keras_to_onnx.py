import onnx
from keras.models import load_model
import onnxmltools
import keras2onnx


# carregando nosso modelo do keras
model = load_model('c:\fer2013_mini_XCEPTION.107-0.66.hdf5')

# convertendo para onnx
onnx_model = onnxmltools.convert_keras(model, target_opset=7)

# salvando em um arquivo
onnx.save(onnx_model, 'modelo_analise_sentimento.onnx')