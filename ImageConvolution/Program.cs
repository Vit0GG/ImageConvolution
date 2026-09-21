using System;
using System.Data;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Threading.Tasks;

using BenchmarkDotNet.Running;

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

#if DEBUG
            Console.WriteLine("Сборка DEBUG. Для замеров используйте: dotnet run -c Release");
#endif
            Console.WriteLine("=== Программа свёртки изображений ===");
            Console.WriteLine("1. Обработать один файл");
            Console.WriteLine("2. Обработать набор файлов");
            Console.WriteLine("3. Обработать набор файлов агентами");
            Console.WriteLine("4. Сравнить с библиотекой ImageSharp");
            Console.WriteLine("5. Обработать файл на GPU");
            Console.WriteLine("6. Создать 4K тестовые изображения");
            Console.WriteLine("7. Unified обработка (CPU + GPU)");
            Console.WriteLine("8. Запустить BenchmarkDotNet");

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
                
                string outputDir = Path.Combine(Path.GetDirectoryName(inputDir) ?? "", 
                    "Processed_Output");
                
                Console.WriteLine("Прогрев...");
                BatchProcessor.ProcessImagesNaiveParallel(
                    inputDir, outputDir + "_Warmup", ParallelStrategy.Sequential);
                
                Console.WriteLine("\n--- ТЕСТ 1: Последовательная свёртка ---");
                var times1 = MeasureMultipleRuns(() =>
                {
                    BatchProcessor.ProcessImagesNaiveParallel(
                        inputDir, outputDir + "_Seq", ParallelStrategy.Sequential);
                }, 3);
                
                Console.WriteLine($"  Среднее: {times1.Average():F1} мс");
                Console.WriteLine($"  Медиана: {CalculateMedian(times1):F1} мс");
                Console.WriteLine($"  Станд. отклонение: {CalculateStdDev(times1):F1} мс");
                
                BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir + "_WarmupRows", ParallelStrategy.ParallelByRows);
                Console.WriteLine("\n--- ТЕСТ 2: Параллелизм по СТРОКАМ (Y) ---");
                var times2 = MeasureMultipleRuns(() =>
                {
                    BatchProcessor.ProcessImagesNaiveParallel(
                        inputDir, outputDir + "_ParRows", ParallelStrategy.ParallelByRows);
                }, 3);
                
                Console.WriteLine($"  Среднее: {times2.Average():F1} мс");
                Console.WriteLine($"  Медиана: {CalculateMedian(times2):F1} мс");
                Console.WriteLine($"  Станд. отклонение: {CalculateStdDev(times2):F1} мс");
                
                BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir + "_WarmupCols", ParallelStrategy.ParallelByColumns);
                Console.WriteLine("\n--- ТЕСТ 3: Параллелизм по СТОЛБЦАМ (X) ---");
                var times3 = MeasureMultipleRuns(() =>
                {
                    BatchProcessor.ProcessImagesNaiveParallel(
                        inputDir, outputDir + "_ParCols", ParallelStrategy.ParallelByColumns);
                }, 3);
                
                Console.WriteLine($"  Среднее: {times3.Average():F1} мс");
                Console.WriteLine($"  Медиана: {CalculateMedian(times3):F1} мс");
                Console.WriteLine($"  Станд. отклонение: {CalculateStdDev(times3):F1} мс");
            }
            else if (choice == "3")
            {
                Console.WriteLine("Введите путь к папке:");
                string? inputDir = Console.ReadLine()?.Trim('\"', ' ', '\'');
                if (string.IsNullOrEmpty(inputDir) || !Directory.Exists(inputDir)) return;

                Console.Write("Введите количество агентов для свёртки (от количества ядер на устройстве): ");
                if (!int.TryParse(Console.ReadLine(), out int workerCount) || workerCount <= 0)
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
LibraryProcessor.ProcessImagesWithImageSharp(inputDir, outputDir);
                });

                Console.WriteLine($"ImageSharp blur (median): {ms:F3} ms");
            }

            else if (choice == "5")
            {
                Console.Write("Введите путь к файлу или папке для обработки на GPU: ");
                string? path = Console.ReadLine()?.Trim('\"', ' ', '\'');

                if (string.IsNullOrEmpty(path)) return;

                if (File.Exists(path))
                {
                    Console.WriteLine("\nРежим: обработка одного файла на GPU.");
                    float[,] img = ImageIO.LoadAsGrayscaleFloat(path);

                    using var gpuProcessor = new GpuConvolutionProcessor();

                    Console.WriteLine("Прогрев GPU...");
                    _ = gpuProcessor.ConvolveGpu(img, Kernels.BlurBoxFloat, EdgeStrategy.Extend);

                    Console.WriteLine("\n--- Замер производительности свёртки на GPU ---");
                    double gpuMs = MeasureMedianMs(() =>
                    {
                        _ = gpuProcessor.ConvolveGpu(img, Kernels.BlurBoxFloat, EdgeStrategy.Extend);
                    }, warmupRuns: 1, measuredRuns: 5);

                    Console.WriteLine($"\nРезультат: время свёртки (медиана): {gpuMs:F3} мс");

                    string dir = Path.GetDirectoryName(path)!;
                    string fname = Path.GetFileNameWithoutExtension(path);
                    string ext = Path.GetExtension(path);
                    string outputPath = Path.Combine(dir, $"{fname}_gpu_processed{ext}");

                    float[,] res = gpuProcessor.ConvolveGpu(img, Kernels.BlurBoxFloat, EdgeStrategy.Extend);
                    ImageIO.SaveImageFloat(res, outputPath);
                    Console.WriteLine($"Результат сохранен в: {outputPath}");
                }
                else if (Directory.Exists(path))
                {
                    Console.WriteLine("\nРежим: пакетная обработка на GPU.");
                    string outputDir = Path.Combine(Path.GetDirectoryName(path)!, "GPU_Batch_Output");

                    Console.WriteLine("Прогрев GPU...");
                    GpuConvolutionProcessor.ProcessDirectory(path, outputDir + "_Warmup", Kernels.BlurBoxFloat, EdgeStrategy.Extend, false);

                    Console.WriteLine("\n--- Замер пакетной обработки на GPU ---");
                    var times = MeasureMultipleRuns(() =>
                    {
                        GpuConvolutionProcessor.ProcessDirectory(path, outputDir, Kernels.BlurBoxFloat, EdgeStrategy.Extend, false);
                    }, 3);

                    Console.WriteLine($"\n  Среднее: {times.Average():F1} мс");
                    Console.WriteLine($"  Медиана: {CalculateMedian(times):F1} мс");
                    Console.WriteLine($"  Станд. отклонение: {CalculateStdDev(times):F1} мс");
                }
            }

            else if (choice == "6")
            {
                string outputDir = Path.Combine(Environment.CurrentDirectory, "Test4K");
                Directory.CreateDirectory(outputDir);

                int count = 100;
                Console.WriteLine($"Создание {count} изображений 4K...");

                for (int i = 0; i < count; i++)
                {
                    double[,] image = new double[2160, 3840];
                    var rand = new Random(i);
                    for (int y = 0; y < 2160; y++)
                    {
                        for (int x = 0; x < 3840; x++)
                        {
                            image[y, x] = rand.Next(0, 256);
                        }
                    }
                    ImageIO.SaveImage(image, Path.Combine(outputDir, $"test_{i:000}.jpg"));
                    Console.WriteLine($"Создано: {i + 1}/{count}");
                }
                Console.WriteLine($"Готово, Папка: {outputDir}");
            }
            else if (choice == "7")
            {
                Console.WriteLine("Введите путь к папке:");
                string? inputDir = Console.ReadLine()?.Trim('\"', ' ', '\'');
                if (string.IsNullOrEmpty(inputDir) || !Directory.Exists(inputDir)) return;

                Console.Write("CPU воркеров (Enter для auto): ");
                string? cpuInput = Console.ReadLine();
                int cpuWorkers = string.IsNullOrEmpty(cpuInput) ? Math.Max(1, Environment.ProcessorCount / 2) : int.TryParse(cpuInput, out int cpuValue) && cpuValue >= 0 ? cpuValue : throw new ArgumentException("Некорректное число CPU агентов");

                Console.Write("GPU воркеров (Enter для 1): ");
                string? gpuInput = Console.ReadLine();
                int gpuWorkers = string.IsNullOrEmpty(gpuInput) ? 1 : int.TryParse(gpuInput, out int gpuValue) && gpuValue >= 0 ? gpuValue : throw new ArgumentException("Некорректное число GPU агентов");

                string outputDir = Path.Combine(Path.GetDirectoryName(inputDir)!, "Unified_Output");

                var config = new UnifiedProcessorConfig
                {
                    CpuWorkers = cpuWorkers,
                    GpuWorkers = gpuWorkers,
                    ReaderThreads = 1,
                    WriterThreads = 1,
                    Kernel = Kernels.BlurBoxFloat,
                    Strategy = EdgeStrategy.Extend
                };

                Console.WriteLine("\nПрогрев...");
                using (var processor = new UnifiedProcessor(config))
                {
                    processor.ProcessDirectory(inputDir, outputDir + "_Warmup");
                }

                Console.WriteLine("\n\n=== Основной запуск ===");
                using (var processor = new UnifiedProcessor(config))
                {
                    processor.ProcessDirectory(inputDir, outputDir);
                }
            }
            else if (choice == "8")
            {
                Console.WriteLine("Внимание: BenchmarkDotNet требует сборки в режиме Release, иначе результаты будут с предупреждением.");
                BenchmarkRunner.Run<ConvolutionBenchmarks>();
            }
            else
            {
                Console.WriteLine("Неверный выбор.");
            }

        }
        static double MeasureMedianMs(Action action, int warmupRuns = 2, int measuredRuns = 5)
        {
            for (int i = 0; i < warmupRuns; i++) action();
            return CalculateMedian(MeasureMultipleRuns(action, measuredRuns));
        }
        static List<double> MeasureMultipleRuns(Action action, int runs)
        {
            var times = new List<double>();

            for (int i = 0; i < runs; i++)
            {

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
        public static byte ToByte(double value) => !(value > 0) ? (byte)0 : value >= 255 ? (byte)255 : (byte)(value + 0.5);


        public static byte[,] LoadAsGrayscaleByte(string path)
        {
            using Image<Rgba32> image = Image.Load<Rgba32>(path);
            byte[,] result = new byte[image.Height, image.Width];
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < image.Height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < image.Width; x++)
                        result[y, x] = ToByte(0.299 * row[x].R + 0.587 * row[x].G + 0.114 * row[x].B);
                }
            });
            return result;
        }

    public static void SaveImageByte(byte[,] data, string path)
    {
        int height = data.GetLength(0);
        int width = data.GetLength(1);
        
        using Image<L8> image = new Image<L8>(width, height);

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < height; y++)
            {
                Span<L8> pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < width; x++)
                {
                    pixelRow[x] = new L8(data[y, x]);
                }
            }
        });
        
        image.Save(path);
    }

    public static double[,] LoadAsGrayscale(string path)
    {
        using Image<Rgba32> image = Image.Load<Rgba32>(path);
        int width = image.Width;
        int height = image.Height;
        double[,] result = new double[height, width];

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < height; y++)
            {
                Span<Rgba32> pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < width; x++)
                {
                    ref Rgba32 pixel = ref pixelRow[x];
                    result[y, x] = 0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B;
                }
            }
        });
        return result;
    }

    public static void SaveImage(double[,] data, string path)
    {
        int height = data.GetLength(0);
        int width = data.GetLength(1);
        using Image<L8> image = new Image<L8>(width, height);

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < height; y++)
            {
                Span<L8> pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < width; x++)
                {
                    double val = data[y, x];
                    byte gray = ToByte(val);
                    pixelRow[x] = new L8(gray);
                }
            }
        });
        image.Save(path);
    }

    public static float[,] LoadAsGrayscaleFloat(string path)
    {
        using Image<Rgba32> image = Image.Load<Rgba32>(path);
        int width = image.Width;
        int height = image.Height;
        float[,] result = new float[height, width];

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < height; y++)
            {
                Span<Rgba32> pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < width; x++)
                {
                    ref Rgba32 pixel = ref pixelRow[x];
                    result[y, x] = (float)(0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B);
                }
            }
        });
        return result;
    }

    public static void SaveImageFloat(float[,] data, string path)
    {
        int height = data.GetLength(0);
        int width = data.GetLength(1);
        using Image<L8> image = new Image<L8>(width, height);

        image.ProcessPixelRows(accessor =>
        {
            for (int y = 0; y < height; y++)
            {
                Span<L8> pixelRow = accessor.GetRowSpan(y);
                for (int x = 0; x < width; x++)
                {
                    float val = data[y, x];
                    byte gray = ToByte(val);
                    pixelRow[x] = new L8(gray);
                }
            }
        });
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
        public static float[,] SharpenFloat = new float[,]
       {
            {  0, -1,  0},
            { -1,  5, -1},
            {  0, -1,  0}
       };
        public static float[,] LaplacianFloat = new float[,]
        {
            {-1, -1, -1},
            {-1,  8, -1},
            {-1, -1, -1}
        };
        public static float[,] BlurBoxFloat = new float[,]
        {
            {1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f},
            {1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f},
            {1.0f/9.0f, 1.0f/9.0f, 1.0f/9.0f}
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
                    int pixelY = y + offsetY - ky;
                    int pixelX = x + offsetX - kx;
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
            return ConvolutionCore.Convolve(image, kernel, strategy, ParallelStrategy.Sequential);
        }
    }
    public class ParallelConvolutionProcessor
    {

        public static double[,] ConvolveParallel(double[,] image, double[,] kernel, EdgeStrategy strategy = EdgeStrategy.Extend)
        {
            return ConvolutionCore.Convolve(image, kernel, strategy, ParallelStrategy.ParallelByRows);
        }

        public static double[,] ConvolveParallelByColumns(double[,] image, double[,] kernel, EdgeStrategy strategy = EdgeStrategy.Extend)
        {
            return ConvolutionCore.Convolve(image, kernel, strategy, ParallelStrategy.ParallelByColumns);
        }

    }
}
