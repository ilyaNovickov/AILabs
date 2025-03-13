using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIModel
{
    /// <summary>
    /// Доп функции для нейростеи
    /// </summary>
    public static class ExtraFuncs
    {
        /// <summary>
        /// RaLu
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static double Relu(double val) => Math.Max(0, val);

        /// <summary>
        /// Производная от ReLu
        /// </summary>
        /// <param name="val"></param>
        /// <returns></returns>
        public static double DivRelu(double val) => val < 0 ? 0d : 1d;

        /// <summary>
        /// SoftMax
        /// </summary>
        /// <param name="t"></param>
        /// <returns></returns>
        public static Matrix<double> SoftMax(Matrix<double> t)
        {
            double sum = t.Row(0).Sum(x => Math.Exp(x));
            return t.Map(x => Math.Exp(x) / sum);
        }

        /// <summary>
        /// Получение горизонтального вектора из одного числа 
        /// </summary>
        /// <param name="value"></param>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static Matrix<double> ValueToMatrix(double value, int dimensions)
        {
            double[] arr = new double[dimensions];
            arr[(int)value] = 1d;
            return Vector<double>.Build.Dense(arr).ToRowMatrix();
        }

        /// <summary>
        /// Функция кросс-энтропии
        /// </summary>
        /// <param name="y"></param>
        /// <param name="z"></param>
        /// <returns></returns>
        public static double CrossEntropia(double y, Matrix<double> z)
        {
            return -Math.Log(z.At(0, (int)y));
        }
    }
}
