using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static System.Runtime.InteropServices.JavaScript.JSType;

namespace AIModel
{
    public class Extramethods
    {
        public static List<string[]> GetHalfNegativeVals(List<string[]> data)
        {
            Random random = new Random();

            data = data.OrderBy(x => random.Next()).ToList();

            List<string[]> negative = new(data.Count / 2);

            for (int i = 0; i <= data.Count / 2; i++)
            {
                List<string> values = data[i].Skip(1).Select((val) => (255 - int.Parse(val)).ToString()).ToList();
                values.Insert(0, data[i][0]);
                negative.Add(values.ToArray());
            }

            List<string[]> res = new(data.Count);

            res.AddRange(negative);
            res.AddRange(data.Skip(data.Count / 2 + 1));
            res = res.OrderBy(x => random.Next()).ToList();

            return res;
        }
    }
}
