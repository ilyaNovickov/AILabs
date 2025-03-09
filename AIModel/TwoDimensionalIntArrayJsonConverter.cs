﻿using System.Text.Json.Serialization;
using System.Text.Json;
using System.Linq;

public class TwoDimensionalIntArrayJsonConverter : JsonConverter<double[,]>
{
    public override double[,]? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        using var jsonDoc = JsonDocument.ParseValue(ref reader);

        var rowLength = jsonDoc.RootElement.GetArrayLength();
        var columnLength = jsonDoc.RootElement.EnumerateArray().First().GetArrayLength();

        double[,] grid = new double[rowLength, columnLength];

        int row = 0;
        foreach (var array in jsonDoc.RootElement.EnumerateArray())
        {
            int column = 0;
            foreach (var number in array.EnumerateArray())
            {
                grid[row, column] = number.GetDouble();
                column++;
            }
            row++;
        }

        return grid;
    }
    public override void Write(Utf8JsonWriter writer, double[,] value, JsonSerializerOptions options)
    {
        writer.WriteStartArray();
        for (int i = 0; i < value.GetLength(0); i++)
        {
            writer.WriteStartArray();
            for (int j = 0; j < value.GetLength(1); j++)
            {
                writer.WriteNumberValue(value[i, j]);
            }
            writer.WriteEndArray();
        }
        writer.WriteEndArray();
    }
}