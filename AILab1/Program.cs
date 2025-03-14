using System.Globalization;
using System.Reflection.Emit;
using AIModel;

namespace AILab1
{
    internal class Program
    {
        static void Main(string[] args)
        {
            Work(args);
            //StartNeura.FileCSV = "mnist_train.csv";

            //StartNeura.RunLearning();

            //Logger.Log($"Время обучения : {StartNeura.Time.ToString("hh\\:mm\\:ss")}");

            //StartNeura.RunTest();
            //"mnist_test.csv";

            
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
    }
}
