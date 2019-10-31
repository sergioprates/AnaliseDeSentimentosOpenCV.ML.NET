using Microsoft.ML.Transforms.Image;
using System.Drawing;

namespace AnaliseDeSentimentosOpenCV.ML.NET.ML.Inputs
{
    public struct InputModeloEmocaoConfiguracoes
    {
        public const int AlturaImagem = 64;
        public const int LarguraImagem = 64;
    }

    public class InputModeloEmocao
    {
        [ImageType(InputModeloEmocaoConfiguracoes.AlturaImagem, InputModeloEmocaoConfiguracoes.LarguraImagem)]
        public Bitmap Imagem { get; set; }
    }
}
