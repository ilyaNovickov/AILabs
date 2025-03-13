using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIModel
{
    /// <summary>
    /// Класс доп информации о событии логирования
    /// </summary>
    public class LogEventArgs : EventArgs
    {
        /// <summary>
        /// Сообщение
        /// </summary>
        public string? Message { get; private set; } = null;

        public LogEventArgs(string? message)
        {
            Message = message;
        }
    }

    /// <summary>
    /// класс логгирования
    /// </summary>
    public class Logger : IDisposable
    {
        public static Logger Instance { get; set; }

        private string? filePath = null;
        private FileStream? stream = null;
        private StreamWriter? sw = null;

        /// <summary>
        /// Путь к файлу
        /// </summary>
        public string? FilePath
        {
            get => filePath;
            set
            {
                filePath = value;
                Dispose();
                if (filePath != null)
                {
                    stream = new FileStream(filePath, FileMode.Create, FileAccess.Write);
                    sw = new StreamWriter(stream);
                }
                else
                {
                    stream = null;
                    sw = null;
                }
            }
        }

        /// <summary>
        /// Событие логирования
        /// </summary>
        public static event EventHandler<LogEventArgs>? LogEvent;

        /// <summary>
        /// Логирование сообщения в файл
        /// </summary>
        /// <param name="data"></param>
        public static void Log(string data)
        {
            Instance._Log(data);
        }

        private void _Log(string data)
        {
            sw?.WriteLine(data);
            LogEvent?.Invoke(null, new LogEventArgs(data));
        }

        /// <summary>
        /// Освобождение ресурсов потоков записи данных в файл
        /// </summary>
        public void Dispose()
        {
            sw?.Dispose();
            stream?.Dispose(); 
        }
    }
}
