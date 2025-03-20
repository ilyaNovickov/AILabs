using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using AIModel;


namespace AILab1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            //Work(args);
            Work2();
        }

        private static void Work(string[] args)
        {
            Logger.Instance = new Logger()
            {
                FilePath = "log.txt"
            };
            Logger.LogEvent += (sender, e) => { Console.WriteLine(e.Message); };
            Console.CancelKeyPress += (sender, e) => { StartNeura.Stop(); };

            for (int i = 0; i < args.Length; i++)
            {
                if (args[i] == "--help")
                {
                    Console.WriteLine("Эталон : \"--learningRate 0.001 --epochCount 20 --path \"mnist_train.csv\" --mode learn --path \"mnist_test\" --mode test\"");
                    break;
                }
                if (args[i] == "--learningRate")
                {
                    i++;
                    StartNeura.LearningRate = double.Parse(args[i], NumberStyles.Number | NumberStyles.AllowDecimalPoint);
                }
                else if (args[i] == "--epochCount")
                {
                    i++;
                    StartNeura.EpochCount = int.Parse(args[i]);
                }
                else if (args[i] == "--path")
                {
                    i++;
                    StartNeura.FileCSV = args[i];
                    continue;
                }
                else if (args[i] == "--mode")
                {
                    i++;
                    switch (args[i])
                    {
                        case "learn":
                            StartNeura.RunLearning();
                            Logger.Log($"Время обучения : {StartNeura.Time.ToString("hh\\:mm\\:ss")}");
                            break;
                        case "test":
                            StartNeura.RunTest();
                            break;
                        default:
                            throw new Exception(">:-\\");
                    }
                }
                
            }


            Logger.Instance.Dispose();
        }

        private static void Work2()
        {
            Logger.Instance = new Logger()
            {
                FilePath = "log.txt"
            };
            Logger.LogEvent += (sender, e) => { Console.WriteLine(e.Message); };
            Console.CancelKeyPress += (sender, e) => { StartNeura.Stop(); };

            const string trainPath = "mnist_train.csv";
            const string testPath = "mnist_test.csv";

            StartNeura.FileCSV = trainPath;
            StartNeura.RunLearning();
            StartNeura.FileCSV = testPath;
            StartNeura.RunTest();

            File.Move("MNIST_TRAIN_E.csv", @"old\" + "MNIST_TRAIN_E.csv", true);
            File.Move("MNIST_TRAIN_Accuraty.csv", @"old\" + "MNIST_TRAIN_Accuraty.csv", true);
            File.Move("MNIST_TEST_E.csv", @"old\" + "MNIST_TEST_E.csv", true);
            File.Move("MNIST_TEST_Accuraty.csv", @"old\" + "MNIST_TEST_Accuraty.csv", true);

            StartNeura.RunTest(negative: true);

            File.Move(StartNeura.SavePath, @"old\" + StartNeura.SavePath, true);
            File.Move("MNIST_TEST_E.csv", @"negativeTest\" + "MNIST_TEST_E.csv", true);
            File.Move("MNIST_TEST_Accuraty.csv", @"negativeTest\" + "MNIST_TEST_Accuraty.csv", true);

            StartNeura.FileCSV = trainPath;
            StartNeura.RunLearning(negative: true);
            StartNeura.FileCSV = testPath;
            StartNeura.RunTest(negative: true);

            Logger.Instance.Dispose();
        }
    }
}
