using AnaliseDeSentimentosOpenCV.ML.NET.ML.Inputs;
using AnaliseDeSentimentosOpenCV.ML.NET.ML.Modelos;
using AnaliseDeSentimentosOpenCV.ML.NET.ML.Outputs;
using Microsoft.ML;
using Microsoft.ML.Transforms.Image;
using System.Collections.Generic;

namespace AnaliseDeSentimentosOpenCV.ML.NET.ML
{
    public class ConfiguracaoModeloOnnx
    {
        private MLContext _mlContext;
        private ITransformer _mlModel;

        public PredictionEngine<InputModeloEmocao, OutputModeloEmocao> InicializarModeloEmocao(ModeloEmocao modeloOnnx)
        {
            _mlContext = new MLContext();

            var dataView = _mlContext.Data.LoadFromEnumerable(new List<InputModeloEmocao>());

            var pipeline = _mlContext.Transforms.ResizeImages(resizing: ImageResizingEstimator.ResizingKind.Fill, outputColumnName: modeloOnnx.InputModelo, imageWidth: InputModeloEmocaoConfiguracoes.LarguraImagem,
                imageHeight: InputModeloEmocaoConfiguracoes.AlturaImagem, inputColumnName: nameof(InputModeloEmocao.Imagem))
            .Append(_mlContext.Transforms.ConvertToGrayscale(outputColumnName: modeloOnnx.InputModelo, inputColumnName: modeloOnnx.InputModelo))
            .Append(_mlContext.Transforms.ExtractPixels(outputColumnName: modeloOnnx.InputModelo, inputColumnName: modeloOnnx.InputModelo, colorsToExtract: ImagePixelExtractingEstimator.ColorBits.Blue, interleavePixelColors: true, outputAsFloatArray: true, scaleImage: .0039216f))
            .Append(_mlContext.Transforms.ApplyOnnxModel(modelFile: modeloOnnx.CaminhoModelo, outputColumnName: modeloOnnx.OutputModelo, inputColumnName: modeloOnnx.InputModelo));

            _mlModel = pipeline.Fit(dataView);

            return _mlContext.Model.CreatePredictionEngine<InputModeloEmocao, OutputModeloEmocao>(_mlModel);
        }
    }
}
