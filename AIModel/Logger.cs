using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AIModel
{
    public class LogEventArgs : EventArgs
    {
        public string? Message { get; private set; } = null;

        public LogEventArgs(string? message)
        {
            Message = message;
        }
    }

    public static class Logger
    {
        private static string? filePath = null;
        private static FileStream? stream = null;
        private static StreamWriter sw = null;

        public static string? FilePath
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

        public static event EventHandler<LogEventArgs>? LogEvent;

        public static void Log(string data)
        {
            sw.WriteLine(data);
            LogEvent?.Invoke(null, new LogEventArgs(data));
        }

        public static void Dispose()
        {
            stream?.Dispose();
            sw?.Dispose();
        }
    }
}
