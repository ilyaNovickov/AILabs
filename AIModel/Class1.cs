﻿using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using System.Text.Json;
using System.IO;

namespace AIModel
{
    internal class Class1
    {
    }



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

            string json = JsonSerializer.Serialize(weights, new JsonSerializerOptions { WriteIndented = true });
            File.WriteAllText(filePath, json);

            Console.WriteLine("✅ Веса успешно сохранены!");
        }

        public static (Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>, Matrix<double>)
       LoadWeights(string filePath)
        {
            if (!File.Exists(filePath))
                throw new FileNotFoundException("Файл с весами не найден!");

            string json = File.ReadAllText(filePath);
            ModelWeights weights = JsonSerializer.Deserialize<ModelWeights>(json);

            Matrix<double> w1 = DenseMatrix.OfArray(weights.W1);
            Matrix<double> b1 = DenseMatrix.OfArray(weights.B1);
            Matrix<double> w2 = DenseMatrix.OfArray(weights.W2);
            Matrix<double> b2 = DenseMatrix.OfArray(weights.B2);
            Matrix<double> w3 = DenseMatrix.OfArray(weights.W3);
            Matrix<double> b3 = DenseMatrix.OfArray(weights.B3);

            Console.WriteLine("✅ Веса успешно загружены!");
            return (w1, b1, w2, b2, w3, b3);
        }
    }

    
}
