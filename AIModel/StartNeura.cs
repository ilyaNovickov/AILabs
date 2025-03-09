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
        public static string? FileCSV
        {
            get => file;
            set => file = value;
        }

        public static TimeSpan Time { get; private set; }

        public static bool UseLearnedNeuro { get; set; }

        public static void RunLearning(bool continueLearning = false)
        {
            double learningRate = 0.001d;

            IEnumerable<string[]> data = CSVHelper.ReadCSV(FileCSV);

            string saveFilePath = "training_state.json";

            Matrix<double> w1 = null, b1 = null, w2 = null, b2 = null, w3 = null, b3 = null;

            if (UseLearnedNeuro)
                (w1, b1, w2, b2, w3, b3) = ModelWeights.LoadWeights("model_weights.json");
            else
            {
                (w1, b1, w2, b2, w3, b3) = NeuralWork.FillRandomValues();
            }

            Matrix<double> w1Clone = w1.Clone();

            List<double> eList = null;
            List<double> accuratyList = null;

            Stopwatch sw = new Stopwatch();
            sw.Start();
            (eList, accuratyList, w1, b1, w2, b2, w3, b3) = NeuralWork.TrainNeuralNetwork(data.ToList<string[]>(), w1, b1, w2, b2, w3, b3, learningRate: 0.001, epochs: 20, saveFilePath, continueLearning);
            sw.Stop();
            Time = sw.Elapsed;

            ModelWeights.SaveWeights("model_weights.json", w1, b1, w2, b2, w3, b3);

            CSVHelper.Write("MNIST_TRAIN_E.csv", eList);
            CSVHelper.Write("MNIST_TRAIN_Accuraty.csv", accuratyList);
        }

        public static void TestNeuraExtra()
        {
            var (w1, b1, w2, b2, w3, b3) = ModelWeights.LoadWeights("model_weights.json");
            string extraFile = "mnist_test.csv";

            IEnumerable<string[]> data = CSVHelper.ReadCSV(extraFile);

            var (eList, accuratyList) = NeuralWork.TestNeuralNetwork(data.ToList<string[]>(), w1, b1, w2, b2, w3, b3);

            CSVHelper.Write("MNIST_TEST_E.csv", eList);
            CSVHelper.Write("MNIST_TEST_Accuraty.csv", accuratyList);
        }

        public static void Stop()
        {
            NeuralWork.PauseTraining();
        }
    }
}
