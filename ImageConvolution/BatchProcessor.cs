using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace ImageConvolution
{
    public enum ParallelStrategy
    {
        Sequential,
        ParallelByRows,
        ParallelByColumns
    }

    public class BatchProcessor
    {
        public static void ProcessImagesNaiveParallel(string inputDirectory, string outputDirectory, ParallelStrategy strategy,
            int maxDegreeOfParallelism = 0, long memoryBudgetBytes = 0)
        {
            ArgumentOutOfRangeException.ThrowIfNegative(maxDegreeOfParallelism);
            ArgumentOutOfRangeException.ThrowIfNegative(memoryBudgetBytes);
            if (!Enum.IsDefined(strategy)) throw new ArgumentOutOfRangeException(nameof(strategy));
            if (!Directory.Exists(inputDirectory)) return;
            ImageFiles.ValidateDirectories(inputDirectory, outputDirectory);
            Directory.CreateDirectory(outputDirectory);
            string[] files = ImageFiles.GetFiles(inputDirectory);

            var options = new ParallelOptions { MaxDegreeOfParallelism = maxDegreeOfParallelism == 0 ? Environment.ProcessorCount : maxDegreeOfParallelism };

            int gen0Start = GC.CollectionCount(0);
            int gen1Start = GC.CollectionCount(1);
            int gen2Start = GC.CollectionCount(2);

            using var process = Process.GetCurrentProcess();
            TimeSpan cpuTimeStart = process.TotalProcessorTime;
            var sw = Stopwatch.StartNew();
            options.MaxDegreeOfParallelism = Math.Min(options.MaxDegreeOfParallelism, ImageFiles.GetInFlightLimit(files, memoryBudgetBytes, 16));
            Console.WriteLine($"Параллельных файлов: {options.MaxDegreeOfParallelism}; свёртка: {strategy}");

            Parallel.ForEach(files, options, currentFile =>
            {
                string fileName = Path.GetFileName(currentFile);
                string savePath = Path.Combine(outputDirectory, fileName);

                double[,] imageData = ImageIO.LoadAsGrayscale(currentFile);
                double[,] result;

                switch (strategy)
                {
                    case ParallelStrategy.Sequential:
                        result = ConvolutionProcessor.Convolve(imageData, Kernels.BlurBox);
                        break;

                    case ParallelStrategy.ParallelByRows:
                        result = ParallelConvolutionProcessor.ConvolveParallel(imageData, Kernels.BlurBox);
                        break;

                    case ParallelStrategy.ParallelByColumns:
                        result = ParallelConvolutionProcessor.ConvolveParallelByColumns(imageData, Kernels.BlurBox);
                        break;

                    default:
                        throw new ArgumentException("Unknown strategy");
                }

                ImageIO.SaveImage(result, savePath);
            });

            sw.Stop();
            TimeSpan cpuTimeEnd = process.TotalProcessorTime;

            int gc0 = GC.CollectionCount(0) - gen0Start;
            int gc1 = GC.CollectionCount(1) - gen1Start;
            int gc2 = GC.CollectionCount(2) - gen2Start;

            double realTimeMs = sw.Elapsed.TotalMilliseconds;
            double cpuTimeMs = (cpuTimeEnd - cpuTimeStart).TotalMilliseconds;

            string strategyName = strategy switch
            {
                ParallelStrategy.Sequential => "Последовательная",
                ParallelStrategy.ParallelByRows => "Параллельная по строкам (Y)",
                ParallelStrategy.ParallelByColumns => "Параллельная по столбцам (X)",
                _ => "Неизвестная"
            };

            Console.WriteLine($"\n--- РЕЗУЛЬТАТЫ ПРОФИЛИРОВАНИЯ ({strategyName}) ---");
            Console.WriteLine($"Реальное время: {realTimeMs:F1} мс");
            Console.WriteLine($"Процессорное время (сумма по всем ядрам): {cpuTimeMs:F1} мс");
            Console.WriteLine($"Сборки мусора (GC): Gen0={gc0}, Gen1={gc1}, Gen2={gc2}");
            Console.WriteLine("----------------------------------------------------------\n");
        }
    }
}
