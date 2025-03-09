using System;
using System.Collections.Generic;
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
using System;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


namespace AIModel
{
    public class NeuralWork
    {
        private static bool pauseRequested = false;

        // Класс для сохранения состояния обучения
        public class TrainingState
        {
            public int LastEpoch { get; set; }
            public double[,] W1 { get; set; }
            public double[,] B1 { get; set; }
            public double[,] W2 { get; set; }
            public double[,] B2 { get; set; }
            public double[,] W3 { get; set; }
            public double[,] B3 { get; set; }
        }

        public static (List<double>, List<double>, Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>) 
            TrainNeuralNetwork(List<string[]> data, Matrix<double> w1, Matrix<double> b1,
                                              Matrix<double> w2, Matrix<double> b2, Matrix<double> w3, Matrix<double> b3,
                                              double learningRate, int epochs, string saveFilePath, bool useSavedData = false)
        {
            Random random = new Random();

            Matrix<double> w1Clone = w1.Clone();

            int startEpoch = 0;
            if (useSavedData)
                startEpoch = LoadTrainingState(ref w1, ref b1, ref w2, ref b2, ref w3, ref b3, saveFilePath);

            Logger.Log($"Начинаем обучение с эпохи {startEpoch + 1}/{epochs}...");

            List<double> accuratyList = new List<double>();
            List<double> e_List = new List<double>();
            List<double> e_Final = new List<double>();

            for (int epoch = startEpoch; epoch < epochs; epoch++)
            {
                Logger.Log($"Эпоха {epoch + 1}/{epochs}...");

                // 🔄 Перемешиваем данные перед каждой эпохой
                data = data.OrderBy(x => random.Next()).ToList();

                // Используем Parallel.ForEach для обработки каждого примера параллельно
                Parallel.ForEach(data, row =>
                {
                    double trueVal = double.Parse(row[0]);
                    List<double> vals = row.Skip(1).Select(val => double.Parse(val) / 255.0).ToList();
                    Matrix<double> inputX = Vector<double>.Build.DenseOfEnumerable(vals).ToRowMatrix();

                    // Прямой проход (forward propagation)
                    Matrix<double> t1 = inputX * w1 + b1;
                    Matrix<double> h1 = t1.Map(ExtraFuncs.Relu);

                    Matrix<double> t2 = h1 * w2 + b2;
                    Matrix<double> h2 = t2.Map(ExtraFuncs.Relu);

                    Matrix<double> t3 = h2 * w3 + b3;
                    Matrix<double> z = ExtraFuncs.SoftMax(t3);
                    double E = ExtraFuncs.CrossEntropia(trueVal, z);

                    // Вычисление ошибки (градиенты)
                    Matrix<double> trueY = ExtraFuncs.ValueToMatrix(trueVal, 10);
                    Matrix<double> dE_dt3 = z - trueY;
                    Matrix<double> dE_dw3 = h2.Transpose() * dE_dt3;
                    Matrix<double> dE_db3 = dE_dt3;

                    Matrix<double> dE_dh2 = dE_dt3 * w3.Transpose();
                    Matrix<double> dE_dt2 = dE_dh2.PointwiseMultiply(t2.Map(ExtraFuncs.DivRelu));
                    Matrix<double> dE_dw2 = h1.Transpose() * dE_dt2;
                    Matrix<double> dE_db2 = dE_dt2;

                    Matrix<double> dE_dh1 = dE_dt2 * w2.Transpose();
                    Matrix<double> dE_dt1 = dE_dh1.PointwiseMultiply(t1.Map(ExtraFuncs.DivRelu));
                    Matrix<double> dE_dw1 = inputX.Transpose() * dE_dt1;
                    Matrix<double> dE_db1 = dE_dt1;

                    // 🔒 Обновление весов с блокировкой
                    lock (w1)
                    {
                        w1 -= learningRate * dE_dw1;
                        b1 -= learningRate * dE_db1;
                        w2 -= learningRate * dE_dw2;
                        b2 -= learningRate * dE_db2;
                        w3 -= learningRate * dE_dw3;
                        b3 -= learningRate * dE_db3;

                        e_List.Add(E);
                    }
                });

                e_Final.Add(e_List.Sum() / e_List.Count);
                accuratyList.Add(CalculateAccuracy(data, w1, b1, w2, b2, w3, b3));

                // Проверка на паузу
                if (pauseRequested)
                {
                    // 💾 Сохранение состояния после каждой эпохи
                    SaveTrainingState(epoch, w1, b1, w2, b2, w3, b3, saveFilePath);
                    Logger.Log($"Эпоха {epoch + 1} завершена и сохранена!");

                    Logger.Log("Обучение приостановлено.");
                    break;
                }
            }

            return (e_Final, accuratyList, w1, b1, w2, b2, w3, b3);
        }

        public static (List<double>, List<double>) TestNeuralNetwork(List<string[]> data, Matrix<double> w1, Matrix<double> b1,
                                              Matrix<double> w2, Matrix<double> b2, Matrix<double> w3, Matrix<double> b3)
        {
            Random random = new Random();


            List<double> accuratyList = new List<double>();
            List<double> e_List = new List<double>();
            List<double> e_Final = new List<double>();


            // 🔄 Перемешиваем данные перед каждой эпохой
            //data = data.OrderBy(x => random.Next()).ToList();

            // Используем Parallel.ForEach для обработки каждого примера параллельно
            Parallel.ForEach(data, row =>
            {
                double trueVal = double.Parse(row[0]);
                List<double> vals = row.Skip(1).Select(val => double.Parse(val) / 255.0).ToList();
                Matrix<double> inputX = Vector<double>.Build.DenseOfEnumerable(vals).ToRowMatrix();

                // Прямой проход (forward propagation)
                Matrix<double> t1 = inputX * w1 + b1;
                Matrix<double> h1 = t1.Map(ExtraFuncs.Relu);

                Matrix<double> t2 = h1 * w2 + b2;
                Matrix<double> h2 = t2.Map(ExtraFuncs.Relu);

                Matrix<double> t3 = h2 * w3 + b3;
                Matrix<double> z = ExtraFuncs.SoftMax(t3);
                double E = ExtraFuncs.CrossEntropia(trueVal, z);

                // Вычисление ошибки (градиенты)
                Matrix<double> trueY = ExtraFuncs.ValueToMatrix(trueVal, 10);
                Matrix<double> dE_dt3 = z - trueY;
                Matrix<double> dE_dw3 = h2.Transpose() * dE_dt3;
                Matrix<double> dE_db3 = dE_dt3;

                Matrix<double> dE_dh2 = dE_dt3 * w3.Transpose();
                Matrix<double> dE_dt2 = dE_dh2.PointwiseMultiply(t2.Map(ExtraFuncs.DivRelu));
                Matrix<double> dE_dw2 = h1.Transpose() * dE_dt2;
                Matrix<double> dE_db2 = dE_dt2;

                Matrix<double> dE_dh1 = dE_dt2 * w2.Transpose();
                Matrix<double> dE_dt1 = dE_dh1.PointwiseMultiply(t1.Map(ExtraFuncs.DivRelu));
                Matrix<double> dE_dw1 = inputX.Transpose() * dE_dt1;
                Matrix<double> dE_db1 = dE_dt1;

                lock (w1)
                {
                    e_List.Add(E);
                }
            });

            e_Final.Add(e_List.Sum() / e_List.Count);
            accuratyList.Add(CalculateAccuracy(data, w1, b1, w2, b2, w3, b3));
            

            return (e_Final, accuratyList);
        }

        // Метод для сохранения состояния обучения
        public static void SaveTrainingState(int epoch, Matrix<double> w1, Matrix<double> b1,
                                             Matrix<double> w2, Matrix<double> b2, Matrix<double> w3, Matrix<double> b3,
                                             string filePath)
        {
            var state = new TrainingState
            {
                LastEpoch = epoch + 1,
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
            string json = JsonSerializer.Serialize(state, options);
            File.WriteAllText(filePath, json);
        }

        // Метод для загрузки состояния обучения
        public static int LoadTrainingState(ref Matrix<double> w1, ref Matrix<double> b1,
                                            ref Matrix<double> w2, ref Matrix<double> b2,
                                            ref Matrix<double> w3, ref Matrix<double> b3, string filePath)
        {
            if (!File.Exists(filePath))
            {
                Console.WriteLine("Файл с весами не найден. Начинаем обучение с 0-й эпохи.");
                return 0;
            }

            string json = File.ReadAllText(filePath);
            var options = new JsonSerializerOptions();
            options.Converters.Add(new TwoDimensionalIntArrayJsonConverter());
            TrainingState state = JsonSerializer.Deserialize<TrainingState>(json, options);

            w1 = DenseMatrix.OfArray(state.W1);
            b1 = DenseMatrix.OfArray(state.B1);
            w2 = DenseMatrix.OfArray(state.W2);
            b2 = DenseMatrix.OfArray(state.B2);
            w3 = DenseMatrix.OfArray(state.W3);
            b3 = DenseMatrix.OfArray(state.B3);

            Logger.Log($"Веса загружены. Продолжаем обучение с эпохи {state.LastEpoch}.");
            return state.LastEpoch;
        }

        // Метод для установки паузы
        public static void PauseTraining()
        {
            pauseRequested = true;
        }

        public static double CalculateAccuracy(IEnumerable<string[]> testData, Matrix<double> w1, Matrix<double> b1,
                                      Matrix<double> w2, Matrix<double> b2, Matrix<double> w3, Matrix<double> b3)
        {
            int correctPredictions = 0;
            int totalSamples = testData.Count();

            foreach (var row in testData)
            {
                double trueLabel = double.Parse(row[0]); // Истинное значение

                // Извлекаем пиксели изображения и нормализуем
                List<double> pixelValues = row.Skip(1).Select(val => double.Parse(val) / 255.0).ToList();
                Matrix<double> inputX = Vector<double>.Build.DenseOfEnumerable(pixelValues).ToRowMatrix();

                // Прямой проход (forward propagation)
                Matrix<double> t1 = inputX * w1 + b1;
                Matrix<double> h1 = t1.Map(ExtraFuncs.Relu);

                Matrix<double> t2 = h1 * w2 + b2;
                Matrix<double> h2 = t2.Map(ExtraFuncs.Relu);

                Matrix<double> t3 = h2 * w3 + b3;
                Matrix<double> output = ExtraFuncs.SoftMax(t3);

                // Индекс с максимальным значением — предсказанное число
                int predictedLabel = output.Row(0).MaximumIndex();

                // Проверяем, правильно ли угадала сеть
                if (predictedLabel == (int)trueLabel)
                {
                    correctPredictions++;
                }
            }

            // Вычисляем точность
            return (double)correctPredictions / totalSamples;
        }

        public static (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>)
            FillRandomValues()
        {
            int n = 10, firstLay = 10;//кол-во нейронов в слое №1

            Matrix<double> w1 = DenseMatrix.Build.Random(28 * 28, n, new ContinuousUniform(-1d, +1d));//матрица весов W
            Matrix<double> b1 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

            int seclayer = 15;//кол-во нейронов в слое №1
            n = 15;

            Matrix<double> w2 = DenseMatrix.Build.Random(w1.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W
            Matrix<double> b2 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

            int thirdLayer = 10;//кол-во нейронов в слое №1
            n = 10;

            Matrix<double> w3 = DenseMatrix.Build.Random(w2.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W
            Matrix<double> b3 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

            return (w1, b1, w2, b2, w3, b3);
        }
    }
}
