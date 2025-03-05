﻿using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;

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
            double E2 = CrossEntropia(1, Matrix<double>.Build.DenseOfArray(new double[,] { { 0.1, 0.7, 0.2 } }));


            Func<double, double> activationFunc = Relu;//функция активации
            Func<double, double> divActivationFunc = DivRelu;//Производная функции активации

            double learningRate = 0.001d;


            IEnumerable<string[]> data =  CSVHelper.ReadCSV(FileCSV);

            string[] row = data.ElementAt<string[]>(0);

            double trueVal = double.Parse(row[0]);//истинное значение картинки

            //Значения пикселей картинки
            List<double> vals = (from val in row select double.Parse(val)/255d).ToList<double>();
            vals.RemoveAt(0);

            Matrix<double> inputX = Vector<double>.Build.DenseOfEnumerable(vals).ToRowMatrix();//Входной вектор X (28*28=784)

            /*Слой №1*/
            int n = 10, firstLay = 10;//кол-во нейронов в слое №1

            Matrix<double> w1 = DenseMatrix.Build.Random(inputX.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

            Matrix<double> b1 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

            Matrix<double> t1 = inputX * w1 + b1;//вектор t

            //Matrix<double> h1 = Matrix.Build.Dense(1, n, (row, col) => activationFunc(t1, col));//Заполнение значений слоя h

            Matrix<double> h1 = t1.Map(activationFunc);
            /*КОНЕЦ Слой №1*/

            /*Слой №2*/
            int seclayer = 15;//кол-во нейронов в слое №1
            n = 15;

            Matrix<double> w2 = DenseMatrix.Build.Random(w1.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

            Matrix<double> b2 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

            Matrix<double> t2 = h1 * w2 + b2;//вектор t
            Matrix<double> h2 =  t2.Map(activationFunc);
            //Matrix<double> h2 = Matrix.Build.Dense(1, n, (row, col) => activationFunc(t1, col));//Заполнение значений слоя h
            /*КОНЕЦ Слой №2*/

            /*Слой №3*/
            int thirdLayer = 10;//кол-во нейронов в слое №1
            n = 10;

            Matrix<double> w3 = DenseMatrix.Build.Random(w2.ColumnCount, n, new ContinuousUniform(-1d, +1d));//матрица весов W

            Matrix<double> b3 = Vector<double>.Build.Random(n, new ContinuousUniform(-1d, +1d)).ToRowMatrix();//матрица смещений b

            Matrix<double> t3 = h2 * w3 + b3;//вектор t

            //Matrix<double> h3 = Matrix.Build.Dense(1, n, (row, col) => activationFunc(t1, col));//Заполнение значений слоя h
            /*КОНЕЦ Слой №3*/

            Matrix<double> z = SoftMax(t3);
            double E = CrossEntropia(trueVal, z);

            //Обучение
            Matrix<double> trueY = ValueToMatrix(trueVal, 10);

            var dE_dt3 = z - trueY;
            //var dE_dw3 = h3

            //Matrix<double> dE_dh1 = 0;
            //Matrix<double> dE_dt1 = dE_dh1.PointwiseMultiply(Vector<double>.Build.Dense(t1.ColumnCount, (colum) => divActivationFunc(t1, colum)).ToRowMatrix());

            //# обратное распространение 
            //            y_vect = to_prob_vect(y, OUT_DIM)
            //                dE_dt2 = z - y_vect
            //                dE_dW2 = h1.T @ dE_dt2
            //                dE_db2 = dE_dt2
            //                dE_dh1 = dE_dt2 @ W2.T
            //                dE_dt1 = dE_dh1 * relu_deriv(t1)
            //                dE_dW1 = x.T @ dE_dt1
            //                dE_db1 = dE_dt1

            Matrix<double> test = Matrix<double>.Build.DenseOfArray(new double[,] { { 10, 5, 3 } }).PointwiseMultiply(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 2, 5} }));
            //Matrix<double> test2 = Matrix<double>.Build.DenseOfArray(new double[,] { { 10, 5, 3 } }).Multiply(Matrix<double>.Build.DenseOfArray(new double[,] { { 1, 2, 5 } }));
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
    }
}
