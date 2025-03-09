//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using MathNet.Numerics.LinearAlgebra;
//using MathNet.Numerics.LinearAlgebra.Storage;
//using MathNet.Numerics.LinearAlgebra.Double;
//using MathNet.Numerics.Distributions;
//using System.Runtime.Serialization.Formatters.Binary;
//using System.Text.Json;
//using System.Text.Json.Serialization.Metadata;
//using System.Xml.Serialization;
//using System.Diagnostics;

//namespace AIModel
//{
//    public static class LabDo
//    {
//        static string? file =
//#if DEBUG
//            "mnist_test.csv";
//#else
//            null;
//#endif

//        public static string? FileCSV
//        {
//            get => file;
//            set => file = value;
//        }

//        public static void foo()
//        {

            

            

            

//            for (int epoch = 0; epoch < 10; epoch++)
//            {
//                for (int i = 0; i < data.Count(); i++)
//                {
//                    #region getData
//                    string[] row = data.ElementAt<string[]>(i);

//                    double trueVal = double.Parse(row[0]);//истинное значение картинки

//                    //Значения пикселей картинки
//                    List<double> vals = (from val in row select double.Parse(val) / 255d).ToList<double>();
//                    vals.RemoveAt(0);
//                    #endregion
//                    #region calculate
//                    Matrix<double> inputX = Vector<double>.Build.DenseOfEnumerable(vals).ToRowMatrix();//Входной вектор X (28*28=784)

//                    /*Слой №1*/
//                    //w1 = DenseMatrix.Build.Random(inputX.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

//                    //b1 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

//                    Matrix<double> t1 = inputX * w1 + b1;//вектор t

//                    //Matrix<double> h1 = Matrix.Build.Dense(1, n, (row, col) => activationFunc(t1, col));//Заполнение значений слоя h

//                    Matrix<double> h1 = t1.Map(activationFunc);
//                    /*КОНЕЦ Слой №1*/

//                    /*Слой №2*/
//                    //w2 = DenseMatrix.Build.Random(w1.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

//                    //b2 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

//                    Matrix<double> t2 = h1 * w2 + b2;//вектор t
//                    Matrix<double> h2 = t2.Map(activationFunc);
//                    //Matrix<double> h2 = Matrix.Build.Dense(1, n, (row, col) => activationFunc(t1, col));//Заполнение значений слоя h
//                    /*КОНЕЦ Слой №2*/

//                    /*Слой №3*/
//                    //w3 = DenseMatrix.Build.Random(w2.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

//                    //b3 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

//                    Matrix<double> t3 = h2 * w3 + b3;//вектор t

//                    //Matrix<double> h3 = Matrix.Build.Dense(1, n, (row, col) => activationFunc(t1, col));//Заполнение значений слоя h
//                    /*КОНЕЦ Слой №3*/
//                    #endregion
//                    #region learning
//                    Matrix<double> z = SoftMax(t3);
//                    //double E = CrossEntropia(trueVal, z);

//                    //Обучение
//                    Matrix<double> trueY = ValueToMatrix(trueVal, 10);

//                    Matrix<double> dE_dt3 = z - trueY;
//                    Matrix<double> dE_dw3 = h2.Transpose() * dE_dt3;
//                    Matrix<double> dE_db3 = dE_dt3.Clone();

//                    Matrix<double> dE_dh2 = dE_dt3 * w3.Transpose();
//                    Matrix<double> dE_dt2 = dE_dh2.PointwiseMultiply(Vector<double>.Build.Dense(t2.ColumnCount,
//                        (colum) => divActivationFunc(t2.At(0, colum))).ToRowMatrix());
//                    Matrix<double> dE_dw2 = h1.Transpose() * dE_dt2;
//                    Matrix<double> dE_db2 = dE_dt2.Clone();

//                    Matrix<double> dE_dh1 = dE_dt2 * w2.Transpose();
//                    Matrix<double> dE_dt1 = dE_dh1.PointwiseMultiply(Vector<double>.Build.Dense(t1.ColumnCount,
//                        (colum) => divActivationFunc(t1.At(0, colum))).ToRowMatrix());
//                    Matrix<double> dE_dw1 = inputX.Transpose() * dE_dt1;
//                    Matrix<double> dE_db1 = dE_dt1.Clone();

//                    w3 = w3 - learningRate * dE_dw3;
//                    b3 = b3 - learningRate * dE_db3;
//                    w2 = w2 - learningRate * dE_dw2;
//                    b2 = b2 - learningRate * dE_db2;
//                    w1 = w1 - learningRate * dE_dw1;
//                    b1 = b1 - learningRate * dE_db1;
//                    #endregion
//                }

//                double accuracy = CalculateAccuracy(data, w1, b1, w2, b2, w3, b3);
//                Console.WriteLine($"Accuracy: {accuracy * 100:F2}%");
//            }
//        }     
//    }
//}