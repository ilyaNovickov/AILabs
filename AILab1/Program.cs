using System.Globalization;
using System.IO;
using System.Linq;
using System.Reflection.Emit;
using System.Text;
using AIModel;
using static System.Runtime.InteropServices.JavaScript.JSType;

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
            const string trainPath = "mnist_train.csv";
            const string testPath = "mnist_test.csv";

            var data = CSVHelper.ReadCSV(testPath);
            IEnumerable<string> val;
            string trueVal = "";
            {
                var data2 = data.ElementAt<string[]>(10);
                trueVal = data2.ElementAt<string>(0);
                val = data2.Skip(1);
            }
            

            StringBuilder sb = new();
            int count = 0;

            for (int i = 0; i < val.Count(); i++)
            {
                sb.Append(val.ElementAt<string>(i) == "0" ? ' ' : '#');
                count++;
                if (count == 28)
                {
                    count = 0;
                    sb.AppendLine();
                }
            }
        }
    }
}
