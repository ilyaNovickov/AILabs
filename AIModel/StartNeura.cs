using MathNet.Numerics.Distributions;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text.Json;
using System.Text.Json.Serialization.Metadata;
using System.Xml.Serialization;
using System.Diagnostics;
using System.Runtime.Intrinsics.X86;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace AIModel
{
    public static class StartNeura
    {
        static string? file =
#if DEBUG
            "mnist_train.csv";
#else
            null;
#endif

        /// <summary>
        /// Файл с данными для обучения
        /// </summary>
        public static string? FileCSV
        {
            get => file;
            set => file = value;
        }

        /// <summary>
        /// Время обучения
        /// </summary>
        public static TimeSpan Time { get; private set; }

        /// <summary>
        /// Использовать обученные нейроны
        /// </summary>
        public static bool UseLearnedNeuro { get; set; }

        public static string? SavePath { get; set; } = "model_weights.json";


        public static double LearningRate { get; set; } = 0.001d;

        public static int EpochCount { get; set; } = 20;

        /// <summary>
        /// Обучение нейростеи
        /// </summary>
        /// <param name="continueLearning">Продолжить обучение сети с случае паузы</param>
        public static void RunLearning(bool negative = false, bool continueLearning = false)
        {
            IEnumerable<string[]> data = CSVHelper.ReadCSV(FileCSV);

            string saveFilePath = "training_state.json";

            Matrix<double> w1 = null, b1 = null, w2 = null, b2 = null, w3 = null, b3 = null;

            if (UseLearnedNeuro)
                (w1, b1, w2, b2, w3, b3) = ModelWeights.LoadWeights(SavePath);
            else
            {
                (w1, b1, w2, b2, w3, b3) = NeuralWork.FillRandomValues();
            }

            data = negative ? Extramethods.GetHalfNegativeVals(data.ToList()) : data;

            List<double> eList = null;
            List<double> accuratyList = null;

            Stopwatch sw = new Stopwatch();
            sw.Start();
            (eList, accuratyList, w1, b1, w2, b2, w3, b3) = 
                NeuralWork.TrainNeuralNetwork(data.ToList<string[]>(), 
                w1, b1, w2, b2, w3, b3, learningRate: LearningRate, epochs: EpochCount, saveFilePath, continueLearning);
            sw.Stop();
            Time = sw.Elapsed;

            ModelWeights.SaveWeights(SavePath, w1, b1, w2, b2, w3, b3);

            CSVHelper.Write("MNIST_TRAIN_E.csv", eList);
            CSVHelper.Write("MNIST_TRAIN_Accuraty.csv", accuratyList);
        }

        /// <summary>
        /// Тестирование оубченной нейростеи
        /// Обученные матрицы считываются с файлов сохранения нейростеи после обучения
        /// </summary>
        public static void RunTest(bool negative = false)
        {
            var (w1, b1, w2, b2, w3, b3) = ModelWeights.LoadWeights("model_weights.json");

            IEnumerable<string[]> data = CSVHelper.ReadCSV(FileCSV);

            data = negative ? Extramethods.GetHalfNegativeVals(data.ToList()) : data;

            var (eList, accuratyList) = NeuralWork.TestNeuralNetwork(data.ToList<string[]>(), w1, b1, w2, b2, w3, b3);

            try
            {
                Logger.Log($"Энтропия : {eList[0]}\nТочность : {accuratyList[0]}");
            }
            catch
            {
                Logger.Log("Пусто?");
            }

            CSVHelper.Write("MNIST_TEST_E.csv", eList);
            CSVHelper.Write("MNIST_TEST_Accuraty.csv", accuratyList);
        }

        /// <summary>
        /// Остановка обучения
        /// </summary>
        public static void Stop()
        {
            NeuralWork.PauseTraining();
        }
    }
}
