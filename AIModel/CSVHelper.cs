namespace AIModel
{
    public class CSVHelper
    {
        /// <summary>
        /// Получение списка строк из CSV файла, где каждый 
        /// элемент разделён запятой 'б'
        /// </summary>
        /// <param name="filePath"></param>
        /// <returns></returns>
        public static IEnumerable<string[]> ReadCSV(string filePath)
        {
            var Lines = File.ReadLines(filePath);

            IEnumerable<string[]> CSV = from line in Lines
                                        select (line.Split(',')).ToArray();

            return CSV;
        }

        /// <summary>
        /// Запись данных в CSV файл (разделитель ';')
        /// </summary>
        /// <param name="path"></param>
        /// <param name="data"></param>
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
