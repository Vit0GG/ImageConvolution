using System;
using System.Data;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Threading.Tasks;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ImageConvolution
{
    public enum EdgeStrategy
    {
        Extend,
        ZeroPadding
    }

    [ExcludeFromCodeCoverage]
    class Program
    {
        static void Main(string[] args)
        {

            Console.WriteLine("=== Программа свёртки изображений ===");
            Console.WriteLine("1. Обработать один файл");
            Console.WriteLine("2. Обработать набор файлов");
            Console.WriteLine("3. Обработать набор файлов агентами");
            Console.WriteLine("4. Сравнить с библиотекой ImageSharp");
            string? choice = Console.ReadLine();

            if (choice == "1")
            {
                Console.Write("Введите путь к изображению: ");
                string? inputPath = Console.ReadLine()?.Trim('\"', ' ', '\'');
                if (!File.Exists(inputPath)) return;

                double[,] img = ImageIO.LoadAsGrayscale(inputPath);

                double seqMs = MeasureMedianMs(() =>
                {
                    _ = ConvolutionProcessor.Convolve(img, Kernels.BlurBox, EdgeStrategy.Extend);
                });

                double parMs = MeasureMedianMs(() =>
                {
                    _ = ParallelConvolutionProcessor.ConvolveParallel(img, Kernels.BlurBox, EdgeStrategy.Extend);
                });

                Console.WriteLine($"Compute only (median): seq={seqMs:F3} ms, parallel={parMs:F3} ms");
                double[,] res = ConvolutionProcessor.Convolve(img, Kernels.BlurBox);

                string directory = Path.GetDirectoryName(inputPath)!;
                string fileNameOnly = Path.GetFileNameWithoutExtension(inputPath);
                string extension = Path.GetExtension(inputPath);

                string baseOutputPath = Path.Combine(directory, $"{fileNameOnly}_filtered");

                int counter = 1;
                string finalOutputPath = $"{baseOutputPath}_{counter}{extension}";

                while (File.Exists(finalOutputPath))
                {
                    counter++;
                    finalOutputPath = $"{baseOutputPath}_{counter}{extension}";
                }

                ImageIO.SaveImage(res, finalOutputPath);
                Console.WriteLine($"\nГотово! Изображение сохранено здесь:\n{finalOutputPath}");
            }
            else if (choice == "2")
            {
                Console.WriteLine("Введите исходный путь к папке или перетащите её");
                string? inputDir = Console.ReadLine()?.Trim('\"', ' ', '\'');

                if (string.IsNullOrEmpty(inputDir)) return;

                string outputDir = Path.Combine(Path.GetDirectoryName(inputDir) ?? "", "Processed_Output");

                Console.WriteLine("Прогрев...");
                BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir + "_Warmup", false);

                Console.WriteLine("\n--- ТЕСТ 1: Только внешний параллелизм (последовательная свёртка) ---");
                var times1 = MeasureMultipleRuns(() =>
                {
                    BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir + "_Seq", false);
                }, 3);

                Console.WriteLine($"  Среднее: {times1.Average():F1} мс");
                Console.WriteLine($"  Медиана: {CalculateMedian(times1):F1} мс");
                Console.WriteLine($"  Станд. отклонение: {CalculateStdDev(times1):F1} мс");

                Console.WriteLine("\n--- ТЕСТ 2: Вложенный параллелизм (параллельная свёртка) ---");
                var times2 = MeasureMultipleRuns(() =>
                {
                    BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir + "_Par", true);
                }, 3);

                Console.WriteLine($"  Среднее: {times2.Average():F1} мс");
                Console.WriteLine($"  Медиана: {CalculateMedian(times2):F1} мс");
                Console.WriteLine($"  Станд. отклонение: {CalculateStdDev(times2):F1} мс");
            }
            else if (choice == "3")
            {
                Console.WriteLine("Введите путь к папке:");
                string? inputDir = Console.ReadLine()?.Trim('\"', ' ', '\'');
                if (string.IsNullOrEmpty(inputDir) || !Directory.Exists(inputDir)) return;

                Console.Write("Введите количество агентов для свёртки (от количества ядер на устройстве): ");
                if (!int.TryParse(Console.ReadLine(), out int workerCount))
                {
                    workerCount = Environment.ProcessorCount;
                }

                string outputDir = Path.Combine(Path.GetDirectoryName(inputDir)!, "Agent_Processed_Output");

                Console.WriteLine("Прогрев...");
                AgentProcessor.ProcessImagesWithAgents(inputDir, outputDir + "_Warmup", workerCount);

                Console.WriteLine($"\n--- Обработка агентами ({workerCount} агентов) ---");
                var times = MeasureMultipleRuns(() =>
                {
                    AgentProcessor.ProcessImagesWithAgents(inputDir, outputDir, workerCount);
                }, 3);

                Console.WriteLine($"  Среднее: {times.Average():F1} мс");
                Console.WriteLine($"  Медиана: {CalculateMedian(times):F1} мс");
                Console.WriteLine($"  Станд. отклонение: {CalculateStdDev(times):F1} мс");
            }
            else if (choice == "4")
            {
                Console.WriteLine("Введите путь к папке:");
                string? inputDir = Console.ReadLine()?.Trim('\"', ' ', '\'');
                if (string.IsNullOrEmpty(inputDir) || !Directory.Exists(inputDir)) return;

                string outputDir = Path.Combine(Path.GetDirectoryName(inputDir)!, "ImageSharp_Output");

                double ms = MeasureMedianMs(() =>
                {
                    if (Directory.Exists(outputDir))
                    {
                        Directory.Delete(outputDir, true);
                    }

                    LibraryProcessor.ProcessImagesWithImageSharp(inputDir, outputDir);
                });

                Console.WriteLine($"ImageSharp blur (median): {ms:F3} ms");
            }
            else
            {
                Console.WriteLine("Неверный выбор.");
            }

        }
        static double MeasureMedianMs(Action action, int warmupRuns = 2, int measuredRuns = 5)
        {
            var currentProcess = Process.GetCurrentProcess();
            var oldPriority = currentProcess.PriorityClass;

            try
            {
                currentProcess.PriorityClass = ProcessPriorityClass.High;

                for (int i = 0; i < warmupRuns; i++)
                {
                    action();
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();
                }

                var times = new List<double>();

                for (int i = 0; i < measuredRuns; i++)
                {
                    GC.Collect();
                    GC.WaitForPendingFinalizers();
                    GC.Collect();

                    long startTimestamp = Stopwatch.GetTimestamp();
                    action();
                    long endTimestamp = Stopwatch.GetTimestamp();

                    double elapsedMs = (endTimestamp - startTimestamp) * 1000.0 / Stopwatch.Frequency;
                    times.Add(elapsedMs);
                }

                return CalculateMedian(times);
            }
            finally
            {
                currentProcess.PriorityClass = oldPriority;
            }
        }
        static List<double> MeasureMultipleRuns(Action action, int runs)
        {
            var times = new List<double>();

            for (int i = 0; i < runs; i++)
            {
                GC.Collect();
                GC.WaitForPendingFinalizers();
                GC.Collect();

                long startTimestamp = Stopwatch.GetTimestamp();
                action();
                long endTimestamp = Stopwatch.GetTimestamp();

                double elapsedMs = (endTimestamp - startTimestamp) * 1000.0 / Stopwatch.Frequency;
                times.Add(elapsedMs);
                Console.WriteLine($"  Прогон {i + 1}: {elapsedMs:F1} мс");
            }

            return times;
        }

        static double CalculateMedian(List<double> values)
        {
            var sorted = values.OrderBy(x => x).ToList();
            int count = sorted.Count;

            if (count % 2 == 0)
            {
                return (sorted[count / 2 - 1] + sorted[count / 2]) / 2.0;
            }
            else
            {
                return sorted[count / 2];
            }
        }

        static double CalculateStdDev(List<double> values)
        {
            double avg = values.Average();
            double sumOfSquares = values.Sum(val => (val - avg) * (val - avg));
            return Math.Sqrt(sumOfSquares / values.Count);
        }

    }

    public class ImageIO
    {
        public static double[,] LoadAsGrayscale(string path)
        {
            using Image<Rgba32> image = Image.Load<Rgba32>(path);
            int width = image.Width;
            int height = image.Height;
            double[,] result = new double[height, width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Rgba32 pixel = image[x, y];
                    double gray = 0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B;
                    result[y, x] = gray;
                }
            }
            return result;
        }
        public static void SaveImage(double[,] data, string path)
        {
            int height = data.GetLength(0);
            int width = data.GetLength(1);

            using Image<Rgba32> image = new Image<Rgba32>(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double val = data[y, x];
                    byte gray = (byte)Math.Clamp(val, 0, 255);
                    image[x, y] = new Rgba32(gray, gray, gray);
                }
            }
            image.Save(path);
        }
    }

    public class Kernels
    {
        public static double[,] Sharpen = new double[,]
        {
            {  0, -1,  0},
            { -1,  5, -1},
            {  0, -1,  0}
        };
        public static double[,] Laplacian = new double[,]
        {
            {-1, -1, -1},
            {-1,  8, -1},
            {-1, -1, -1}
        };
        public static double[,] BlurBox = new double[,]
        {
            {1.0/9.0, 1.0/9.0, 1.0/9.0},
            {1.0/9.0, 1.0/9.0, 1.0/9.0},
            {1.0/9.0, 1.0/9.0, 1.0/9.0}
        };
    }

    public class ConvolutionProcessor
    {
        internal static double CalculatePixelValue(double[,] image, double[,] kernel, int x, int y, EdgeStrategy strategy)
        {
            double sum = 0.0;
            int imgheight = image.GetLength(0);
            int imgwidth = image.GetLength(1);
            int kheight = kernel.GetLength(0);
            int kwidth = kernel.GetLength(1);
            int offsetY = kheight / 2;
            int offsetX = kwidth / 2;

            for (int ky = 0; ky < kheight; ky++)
            {
                for (int kx = 0; kx < kwidth; kx++)
                {
                    int pixelY = y + ky - offsetY;
                    int pixelX = x + kx - offsetX;
                    double pixelValue = 0.0;

                    if (strategy == EdgeStrategy.Extend)
                    {
                        pixelY = Math.Clamp(pixelY, 0, imgheight - 1);
                        pixelX = Math.Clamp(pixelX, 0, imgwidth - 1);
                        pixelValue = image[pixelY, pixelX];
                    }
                    else if (strategy == EdgeStrategy.ZeroPadding)
                    {
                        if (pixelY >= 0 && pixelY < imgheight && pixelX >= 0 && pixelX < imgwidth)
                        {
                            pixelValue = image[pixelY, pixelX];
                        }
                    }
                    sum += pixelValue * kernel[ky, kx];
                }
            }
            return sum;
        }

        public static double[,] Convolve(double[,] image, double[,] kernel, EdgeStrategy strategy = EdgeStrategy.Extend)
        {
            int imgheight = image.GetLength(0);
            int imgwidth = image.GetLength(1);
            double[,] result = new double[imgheight, imgwidth];
            for (int y = 0; y < imgheight; y++)
            {
                for (int x = 0; x < imgwidth; x++)
                {
                    result[y, x] = CalculatePixelValue(image, kernel, x, y, strategy);
                }
            }
            return result;
        }
    }
    public class ParallelConvolutionProcessor
    {

        public static double[,] ConvolveParallel(double[,] image, double[,] kernal, EdgeStrategy strategy = EdgeStrategy.Extend)
        {
            int imgheight = image.GetLength(0);
            int imgwidth = image.GetLength(1);
            double[,] result = new double[imgheight, imgwidth];

            Parallel.For(0, imgheight, y =>
            {
                for (int x = 0; x < imgwidth; x++)
                {
                    result[y, x] = ConvolutionProcessor.CalculatePixelValue(image, kernal, x, y, strategy);
                }
            });
            return result;
        }
    }
}