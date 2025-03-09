using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIModel
{
    public class ExtraFuncs
    {
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

        public static double CrossEntropia(double y, Matrix<double> z)
        {
            return -Math.Log(z.At(0, (int)y));
        }
    }
}
