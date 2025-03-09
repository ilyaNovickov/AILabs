using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System.Text.Json;

namespace AIModel
{

    // Класс для хранения весов
    public class ModelWeights
    {
        public double[,] W1 { get; set; }
        public double[,] B1 { get; set; }
        public double[,] W2 { get; set; }
        public double[,] B2 { get; set; }
        public double[,] W3 { get; set; }
        public double[,] B3 { get; set; }

        // Метод сохранения весов
        public static void SaveWeights(string filePath, Matrix<double> w1, Matrix<double> b1,
                                       Matrix<double> w2, Matrix<double> b2, Matrix<double> w3, Matrix<double> b3)
        {
            var weights = new ModelWeights
            {
                W1 = w1.ToArray(),
                B1 = b1.ToArray(),
                W2 = w2.ToArray(),
                B2 = b2.ToArray(),
                W3 = w3.ToArray(),
                B3 = b3.ToArray()
            };

            var options = new JsonSerializerOptions();
            options.WriteIndented = true;
            options.Converters.Add(new TwoDimensionalIntArrayJsonConverter());
            string json = JsonSerializer.Serialize(weights, options);
            File.WriteAllText(filePath, json);

            Logger.Log("Веса успешно сохранены!");
        }

        public static (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>)
       LoadWeights(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("Файл с весами не найден!");

            string json = File.ReadAllText(filePath);
            var options = new JsonSerializerOptions();
            options.Converters.Add(new TwoDimensionalIntArrayJsonConverter());
            ModelWeights weights = JsonSerializer.Deserialize<ModelWeights>(json, options);

            Matrix<double> w1 = DenseMatrix.OfArray(weights.W1);
            Matrix<double> b1 = DenseMatrix.OfArray(weights.B1);
            Matrix<double> w2 = DenseMatrix.OfArray(weights.W2);
            Matrix<double> b2 = DenseMatrix.OfArray(weights.B2);
            Matrix<double> w3 = DenseMatrix.OfArray(weights.W3);
            Matrix<double> b3 = DenseMatrix.OfArray(weights.B3);

            Logger.Log("Веса успешно загружены!");
            return (w1, b1, w2, b2, w3, b3);
        }
    }


}
