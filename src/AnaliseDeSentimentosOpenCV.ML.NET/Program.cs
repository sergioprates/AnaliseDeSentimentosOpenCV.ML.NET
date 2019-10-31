using AnaliseDeSentimentosOpenCV.ML.NET.ML;
using AnaliseDeSentimentosOpenCV.ML.NET.ML.Inputs;
using AnaliseDeSentimentosOpenCV.ML.NET.ML.Modelos;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using System;
using System.Linq;

namespace AnaliseDeSentimentosOpenCV.ML.NET
{
    class Program
    {
        static void Main(string[] args)
        {
            var configModeloEmocao = new ConfiguracaoModeloOnnx();
            
            var modelo = configModeloEmocao.InicializarModeloEmocao(new ModeloEmocao(Environment.CurrentDirectory + @"\Arquivos\modelo_analise_sentimento_v2.onnx"));

            VideoCapture capture = new VideoCapture(0);
            using (Window window = new Window("Camera"))
            using (Mat imagem = new Mat())
            {
                capture.Read(imagem);
                while (true)
                {
                    capture.Read(imagem);

                    if (imagem.Empty()) break;

                    Mat gray = imagem.CvtColor(ColorConversionCodes.BGR2GRAY, 1);
                   

                    var faces = DetectarFace(gray);

                    foreach (var face in faces)
                    {                       
                        var coordenadaAjustada = new Rect(face.X, face.Y, face.Width, face.Height);

                        string sentimento = string.Empty;

                        using (var faceCortada = new Mat(imagem, coordenadaAjustada))
                        {
                            var scoresSentimento = modelo.Predict(new InputModeloEmocao
                            {
                                Imagem = faceCortada.ToBitmap()
                            }).PredictedLabels.ToList();

                            var indiceSentimento = scoresSentimento.IndexOf(scoresSentimento.Max());

                            sentimento = LabelsSentimento[indiceSentimento];
                        }

                        Cv2.Rectangle(imagem,
                           new Point(face.X, face.Y),
                           new Point(face.X + face.Width, face.Y + face.Height),
                           new Scalar(255, 255, 255), 2);

                        Cv2.PutText(imagem, sentimento, new Point(face.X, face.Y - 20), HersheyFonts.HersheyComplex, 1, Scalar.Green);
                    }

                    window.ShowImage(imagem);

                    Cv2.WaitKey(30);
                }
            }
        }

        static string[] LabelsSentimento => new string[]
      {
           "RAIVA", "DESGOSTO", "MEDO", "FELIZ", "TRISTE",
           "SURPRESO", "NEUTRO"
      };

        static Rect[] DetectarFace(Mat imagem)
        {
            var haarCascade = new CascadeClassifier(Environment.CurrentDirectory + @"\arquivos\haarcascade_frontalface_default.xml");
            // Detect faces
            return haarCascade.DetectMultiScale(imagem, 1.3, 5);
        }
    }
}
