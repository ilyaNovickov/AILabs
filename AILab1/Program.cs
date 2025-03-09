using System.Reflection.Emit;
using AIModel;

namespace AILab1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Logger.FilePath = "log.txt";
            Logger.LogEvent += (sender, e) => { Console.WriteLine(e.Message); };

            Console.CancelKeyPress += Console_CancelKeyPress;

            StartNeura.FileCSV = "mnist_train.csv";

            StartNeura.RunLearning();

            Logger.Log($"Время обучения : {StartNeura.Time.ToString("hh\\:mm\\:ss")}");

            StartNeura.TestNeuraExtra();
        }

        private static void Console_CancelKeyPress(object? sender, ConsoleCancelEventArgs e)
        {
            StartNeura.Stop();

        }
    }
}
