using Microsoft.ML.Data;

namespace AnaliseDeSentimentosOpenCV.ML.NET.ML.Outputs
{
    public class OutputModeloEmocao
    {
        [ColumnName("predictions")]
        public float[] PredictedLabels { get; set; }
    }
}
