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

namespace AIModel
{
    public static class LabDo
    {
        static string? file =
#if DEBUG
            "mnist_test.csv";
#else
            null;
#endif

        public static string? FileCSV
        {
            get => file;
            set => file = value;
        }

        public static void foo()
        {
            //double E2 = CrossEntropia(1, Matrix<double>.Build.DenseOfArray(new double[,] { { 0.1, 0.7, 0.2 } }));

            //!!!var (w1, b1, w2, b2, w3, b3) = LoadWeights("model_weights.json");

            #region chooseFuncs
            Func<double, double> activationFunc = Relu;//функция активации
            Func<double, double> divActivationFunc = DivRelu;//Производная функции активации
            #endregion

            double learningRate = 0.001d;

            IEnumerable<string[]> data =  CSVHelper.ReadCSV(FileCSV);

            Stopwatch sw = new Stopwatch();
            sw.Start();

            #region layersValues
            int n = 10, firstLay = 10;//кол-во нейронов в слое №1

            Matrix<double> w1 = DenseMatrix.Build.Random(28*28, n, new ContinuousUniform(-1d, +1d));//матрица весов W

            Matrix<double> b1 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

            int seclayer = 15;//кол-во нейронов в слое №1
            n = 15;

            Matrix<double> w2 = DenseMatrix.Build.Random(w1.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

            Matrix<double> b2 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

            int thirdLayer = 10;//кол-во нейронов в слое №1
            n = 10;

            Matrix<double> w3 = DenseMatrix.Build.Random(w2.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

            Matrix<double> b3 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b
            #endregion

            for (int epoch = 0; epoch < 10; epoch++)
            {
                for (int i = 0; i < data.Count(); i++)
                {
                    #region getData
                    string[] row = data.ElementAt<string[]>(i);

                    double trueVal = double.Parse(row[0]);//истинное значение картинки

                    //Значения пикселей картинки
                    List<double> vals = (from val in row select double.Parse(val) / 255d).ToList<double>();
                    vals.RemoveAt(0);
                    #endregion
                    #region calculate
                    Matrix<double> inputX = Vector<double>.Build.DenseOfEnumerable(vals).ToRowMatrix();//Входной вектор X (28*28=784)

                    /*Слой №1*/
                    //w1 = DenseMatrix.Build.Random(inputX.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

                    //b1 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

                    Matrix<double> t1 = inputX * w1 + b1;//вектор t

                    //Matrix<double> h1 = Matrix.Build.Dense(1, n, (row, col) => activationFunc(t1, col));//Заполнение значений слоя h

                    Matrix<double> h1 = t1.Map(activationFunc);
                    /*КОНЕЦ Слой №1*/

                    /*Слой №2*/
                    //w2 = DenseMatrix.Build.Random(w1.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

                    //b2 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

                    Matrix<double> t2 = h1 * w2 + b2;//вектор t
                    Matrix<double> h2 = t2.Map(activationFunc);
                    //Matrix<double> h2 = Matrix.Build.Dense(1, n, (row, col) => activationFunc(t1, col));//Заполнение значений слоя h
                    /*КОНЕЦ Слой №2*/

                    /*Слой №3*/
                    //w3 = DenseMatrix.Build.Random(w2.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

                    //b3 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

                    Matrix<double> t3 = h2 * w3 + b3;//вектор t

                    //Matrix<double> h3 = Matrix.Build.Dense(1, n, (row, col) => activationFunc(t1, col));//Заполнение значений слоя h
                    /*КОНЕЦ Слой №3*/
                    #endregion
                    #region learning
                    Matrix<double> z = SoftMax(t3);
                    double E = CrossEntropia(trueVal, z);

                    //Обучение
                    Matrix<double> trueY = ValueToMatrix(trueVal, 10);

                    Matrix<double> dE_dt3 = z - trueY;
                    Matrix<double> dE_dw3 = h2.Transpose() * dE_dt3;
                    Matrix<double> dE_db3 = dE_dt3.Clone();

                    Matrix<double> dE_dh2 = dE_dt3 * w3.Transpose();
                    Matrix<double> dE_dt2 = dE_dh2.PointwiseMultiply(Vector<double>.Build.Dense(t2.ColumnCount,
                        (colum) => divActivationFunc(t2.At(0, colum))).ToRowMatrix());
                    Matrix<double> dE_dw2 = h1.Transpose() * dE_dt2;
                    Matrix<double> dE_db2 = dE_dt2.Clone();

                    Matrix<double> dE_dh1 = dE_dt2 * w2.Transpose();
                    Matrix<double> dE_dt1 = dE_dh1.PointwiseMultiply(Vector<double>.Build.Dense(t1.ColumnCount,
                        (colum) => divActivationFunc(t1.At(0, colum))).ToRowMatrix());
                    Matrix<double> dE_dw1 = inputX.Transpose() * dE_dt1;
                    Matrix<double> dE_db1 = dE_dt1.Clone();

                    w3 = w3 - learningRate * dE_dw3;
                    b3 = b3 - learningRate * dE_db3;
                    w2 = w2 - learningRate * dE_dw2;
                    b2 = b2 - learningRate * dE_db2;
                    w1 = w1 - learningRate * dE_dw1;
                    b1 = b1 - learningRate * dE_db1;
                    #endregion
                }

                double accuracy = CalculateAccuracy(data, w1, b1, w2, b2, w3, b3);
                Console.WriteLine($"Accuracy: {accuracy * 100:F2}%");
            }

            //!!!SaveWeights("model_weights.json", w1, b1, w2, b2, w3, b3);

            sw.Stop();
            int stopper = 1;
        }

        //[Obsolete]
        //public static double Relu(Matrix<double> m, int val)
        //{
        //    return Math.Max(0, m.At(0, val));
        //}

        //[Obsolete]
        //public static double DivRelu(Matrix<double> m, int val)
        //{
        //    return m.At(0, val) < 0 ? 0d : 1d;
        //}

        public static double Relu(double val)
        {
            return Math.Max(0, val);
        }

        public static double DivRelu(double val)
        {
            return val < 0 ? 0d : 1d;
        }

        public static Matrix<double> SoftMax(Matrix<double> t)
        {
            double summ = 0d;

            for (int i = 0; i < t.ColumnCount; i++)
            {
                summ += Math.Exp(t.At(0, i));
            }

            return Matrix<double>.Build.Dense(1, t.ColumnCount, 
                (row, col) => Math.Exp(t.At(0, col)) / summ  );
        }

        public static Matrix<double> ValueToMatrix(double value, int dimensions)
        {
            if (value >= dimensions || value < 0 || dimensions <= 0)
                throw new Exception("!!!");

            double[] arr = new double[dimensions];

            arr[(int)value] = 1d;

            return Vector<double>.Build.Dense(arr).ToRowMatrix();
        }

        //public static double CrossEntropia(Matrix<double> y, Matrix<double> z)
        //{
        //    if (y.ColumnCount != z.ColumnCount)
        //        throw new Exception("!!!!");

        //    double summ = 0d;

        //    for (int i = 0; i < y.ColumnCount; i++)
        //    {
        //        summ += (y.At(1, i) * Math.Log(z.At(1, i)));
        //    }

        //    return summ;
        //}

        public static double CrossEntropia(double y, Matrix<double> z)
        {
            return -Math.Log(z.At(0, (int)y));
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
                Matrix<double> h1 = t1.Map(Relu);

                Matrix<double> t2 = h1 * w2 + b2;
                Matrix<double> h2 = t2.Map(Relu);

                Matrix<double> t3 = h2 * w3 + b3;
                Matrix<double> output = SoftMax(t3);

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
    }
}
/*
using System;
using System.Collections.Concurrent;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

public static class ParallelTraining
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

    public static void TrainNeuralNetwork(List<string[]> data, Matrix<double> w1, Matrix<double> b1,
                                          Matrix<double> w2, Matrix<double> b2, Matrix<double> w3, Matrix<double> b3,
                                          double learningRate, int epochs, string saveFilePath)
    {
        Random random = new Random();
        int startEpoch = LoadTrainingState(ref w1, ref b1, ref w2, ref b2, ref w3, ref b3, saveFilePath);

        Console.WriteLine($"Начинаем обучение с эпохи {startEpoch + 1}/{epochs}...");

        for (int epoch = startEpoch; epoch < epochs; epoch++)
        {
            Console.WriteLine($"Эпоха {epoch + 1}/{epochs}...");

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
                Matrix<double> h1 = t1.Map(Relu);

                Matrix<double> t2 = h1 * w2 + b2;
                Matrix<double> h2 = t2.Map(Relu);

                Matrix<double> t3 = h2 * w3 + b3;
                Matrix<double> z = SoftMax(t3);

                // Вычисление ошибки (градиенты)
                Matrix<double> trueY = ValueToMatrix(trueVal, 10);
                Matrix<double> dE_dt3 = z - trueY;
                Matrix<double> dE_dw3 = h2.Transpose() * dE_dt3;
                Matrix<double> dE_db3 = dE_dt3;

                Matrix<double> dE_dh2 = dE_dt3 * w3.Transpose();
                Matrix<double> dE_dt2 = dE_dh2.PointwiseMultiply(t2.Map(DivRelu));
                Matrix<double> dE_dw2 = h1.Transpose() * dE_dt2;
                Matrix<double> dE_db2 = dE_dt2;

                Matrix<double> dE_dh1 = dE_dt2 * w2.Transpose();
                Matrix<double> dE_dt1 = dE_dh1.PointwiseMultiply(t1.Map(DivRelu));
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
                }
            });

            // 💾 Сохранение состояния после каждой эпохи
            SaveTrainingState(epoch, w1, b1, w2, b2, w3, b3, saveFilePath);
            Console.WriteLine($"✅ Эпоха {epoch + 1} завершена и сохранена!");

            // Проверка на паузу
            if (pauseRequested)
            {
                Console.WriteLine("⏸ Обучение приостановлено.");
                break;
            }
        }
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

        string json = JsonSerializer.Serialize(state, new JsonSerializerOptions { WriteIndented = true });
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
        TrainingState state = JsonSerializer.Deserialize<TrainingState>(json);

        w1 = DenseMatrix.OfArray(state.W1);
        b1 = DenseMatrix.OfArray(state.B1);
        w2 = DenseMatrix.OfArray(state.W2);
        b2 = DenseMatrix.OfArray(state.B2);
        w3 = DenseMatrix.OfArray(state.W3);
        b3 = DenseMatrix.OfArray(state.B3);

        Console.WriteLine($"✅ Веса загружены. Продолжаем обучение с эпохи {state.LastEpoch}.");
        return state.LastEpoch;
    }

    // Метод для установки паузы
    public static void PauseTraining()
    {
        pauseRequested = true;
    }

    public static double Relu(double val) => Math.Max(0, val);
    public static double DivRelu(double val) => val < 0 ? 0d : 1d;

    public static Matrix<double> SoftMax(Matrix<double> t)
    {
        double sum = t.Row(0).Sum(x => Math.Exp(x));
        return t.Map(x => Math.Exp(x) / sum);
    }

    public static Matrix<double> ValueToMatrix(double value, int dimensions)
    {
        double[] arr = new double[dimensions];
        arr[(int)value] = 1d;
        return Vector<double>.Build.Dense(arr).ToRowMatrix();
    }
}


📌 Как использовать этот код?

string saveFilePath = "training_state.json";
ParallelTraining.TrainNeuralNetwork(data, w1, b1, w2, b2, w3, b3, learningRate: 0.001, epochs: 10, saveFilePath);


Если обучение нужно приостановить, вызовите:

ParallelTraining.PauseTraining();
 */