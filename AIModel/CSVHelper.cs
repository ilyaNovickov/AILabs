namespace AIModel
{
    public class CSVHelper
    {
        //Список строк строк файла CSV; кадлый элемент - строка разделённая запятой ',' 
        public static IEnumerable<string[]> ReadCSV(string filePath)
        {
            var Lines = File.ReadLines(filePath);

            IEnumerable<string[]> CSV = from line in Lines
                                        select (line.Split(',')).ToArray();

            return CSV;
        }

        public static void Write(string path, IEnumerable<double> data)
        {
            using (FileStream s = new FileStream(path, FileMode.Create, FileAccess.Write))
            using (StreamWriter sw = new StreamWriter(s))
            {
                sw.WriteLine("Index;Val");

                for (int i = 0; i < data.Count(); i++)
                {
                    sw.WriteLine($"{i};{data.ElementAt<double>(i)}");
                }
            }
        }
    }
}
