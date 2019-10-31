import onnx

onnx_model = onnx.load('./modelo/modelo_analise_sentimento.onnx')

endpoint_names = ['predictions/Softmax:0']

for i in range(len(onnx_model.graph.node)):
	for j in range(len(onnx_model.graph.node[i].input)):
		if onnx_model.graph.node[i].input[j] in endpoint_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].input[j] = onnx_model.graph.node[i].input[j].split(':')[0].replace('/Softmax', '')

	for j in range(len(onnx_model.graph.node[i].output)):
		if onnx_model.graph.node[i].output[j] in endpoint_names:
			print('-'*60)
			print(onnx_model.graph.node[i].name)
			print(onnx_model.graph.node[i].input)
			print(onnx_model.graph.node[i].output)

			onnx_model.graph.node[i].output[j] = onnx_model.graph.node[i].output[j].split(':')[0].replace('/Softmax', '')

for i in range(len(onnx_model.graph.input)):
	if onnx_model.graph.input[i].name in endpoint_names:
		print('-'*60)
		print(onnx_model.graph.input[i])
		onnx_model.graph.input[i].name = onnx_model.graph.input[i].name.split(':')[0].replace('/Softmax', '')

for i in range(len(onnx_model.graph.output)):
	if onnx_model.graph.output[i].name in endpoint_names:
		print('-'*60)
		print(onnx_model.graph.output[i])
		onnx_model.graph.output[i].name = onnx_model.graph.output[i].name.split(':')[0].replace('/Softmax', '')

onnx.save(onnx_model, './modelo/modelo_analise_sentimento_v2.onnx')