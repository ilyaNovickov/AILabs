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
    }
}
