import onnx
from keras.models import load_model
import onnxmltools
import keras2onnx
import tensorflow as tf

print(tf.version.VERSION)
# 1.14.0


# carregando nosso modelo do keras
model = load_model('./modelo/fer2013_mini_XCEPTION.107-0.66.hdf5')

# convertendo para onnx
onnx_model = onnxmltools.convert_keras(model, target_opset=7)

# salvando em um arquivo
onnx.save(onnx_model, './modelo/modelo_analise_sentimento.onnx')