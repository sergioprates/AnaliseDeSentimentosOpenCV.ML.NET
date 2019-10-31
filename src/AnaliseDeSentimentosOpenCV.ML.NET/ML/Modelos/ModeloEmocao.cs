namespace AnaliseDeSentimentosOpenCV.ML.NET.ML.Modelos
{
    public class ModeloEmocao
    {
        public ModeloEmocao(string caminhoModelo)
        {
            CaminhoModelo = caminhoModelo;
        }

        public string InputModelo => "input_1";

        public string OutputModelo => "predictions";

        public string CaminhoModelo { get; private set; }
    }
}
