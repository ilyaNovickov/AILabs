using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Storage;
using MathNet.Numerics.LinearAlgebra.Double;

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
            Func<Matrix<double>, int, double> activationFunc = Relu;//функция активации


            IEnumerable<string[]> data =  CSVHelper.ReadCSV(FileCSV);

            string[] row = data.ElementAt<string[]>(0);

            double trueVal = double.Parse(row[0]);//истинное значение картинки

            //Значения пикселей картинки
            List<double> vals = (from val in row select double.Parse(val)).ToList<double>();
            vals.RemoveAt(0);

            Matrix<double> inputX = Vector<double>.Build.DenseOfEnumerable(vals).ToRowMatrix();//Входной вектор X (28*28=784)

            /*Слой №1*/
            int n = 10, firstLay = 10;//кол-во нейронов в слое №1

            Matrix<double> w = DenseMatrix.Build.Random(inputX.ColumnCount, n);//матрица весов W

            Matrix<double> b = Vector<double>.Build.Random(n).ToRowMatrix();//матрица смещений b

            Matrix<double> t = inputX * w + b;//вектор t

            Matrix<double> h = Matrix.Build.Dense(1, n, (row, col) => activationFunc(t, col));//Заполнение значений слоя h
            /*КОНЕЦ Слой №1*/

            var z = SoftMax(h);
            var z2 = SoftMax(Matrix<double>.Build.DenseOfArray(new double[,] { { 1.3d, 5.1d, 2.2d, 0.7d, 1.1d} }));
        }

        public static double Relu(Matrix<double> m, int val)
        {
            return Math.Max(0, m.At(0, val));
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
    }
}
