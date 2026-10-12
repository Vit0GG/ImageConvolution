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
    public enum EdgeStrategy { Extend, ZeroPadding }

    [ExcludeFromCodeCoverage]
    class Program
    {
        static string? ReadPath() => Console.ReadLine()?.Trim('"', ' ', '\'');

        static int ReadCount(string prompt, int fallback, bool allowZero = false)
        {
            Console.Write(prompt);
            string? value = Console.ReadLine();
            if (string.IsNullOrWhiteSpace(value)) return fallback;
            if (!int.TryParse(value, out int count) || count < (allowZero ? 0 : 1))
                throw new ArgumentException("Некорректное количество агентов");
            return count;
        }

        static string OutputFolder(string input, string name)
        {
            string? root = Environment.GetEnvironmentVariable("IMAGECONV_OUTPUT_ROOT");
            return Path.Combine(string.IsNullOrWhiteSpace(root)
                ? Path.GetDirectoryName(Path.TrimEndingDirectorySeparator(Path.GetFullPath(input))) ?? Environment.CurrentDirectory
                : Path.GetFullPath(root), name);
        }

        static void Main(string[] args)
        {
            if (args.Length > 0 && args[0] == "--byte-layout")
            {
                ByteLayoutExperiment.Run(args);
                return;
            }
            if (args.Length > 0 && args[0] == "--locality-parallel-control")
            {
                if (args.Length < 2 || args.Length > 4 || !File.Exists(args[1]))
                    throw new ArgumentException("--locality-parallel-control путь_к_файлу [число_разделов] [число_измерений]");
                int partitions = args.Length > 2 ? int.Parse(args[2]) : Environment.ProcessorCount;
                int runs = args.Length > 3 ? int.Parse(args[3]) : 7;
                LocalityExperiment.ParallelControl(args[1], partitions, runs);
                return;
            }
            if (args.Length > 0 && args[0] == "--locality-profile")
            {
                LocalityExperiment.HardwareWorkload(args);
                return;
            }
#if DEBUG
            Console.WriteLine("Сборка DEBUG. Для замеров используйте: dotnet run -c Release");
#endif
            Console.WriteLine("=== Программа свёртки изображений ===");
            Console.WriteLine("1. Обработать один файл");
            Console.WriteLine("2. Обработать набор файлов");
            Console.WriteLine("3. Обработать набор файлов агентами");
            Console.WriteLine("4. Сравнить с библиотекой ImageSharp");
            Console.WriteLine("5. Обработать файл или папку на GPU");
            Console.WriteLine("6. Создать 4K тестовые изображения");
            Console.WriteLine("7. Unified обработка (CPU + GPU)");
            Console.WriteLine("8. Запустить BenchmarkDotNet");
            Console.WriteLine("9. ImageSharp напрямую: без промежуточных массивов яркости");
            Console.WriteLine("10. Контроль обхода: строки, столбцы, блоки (один файл в памяти)");
            Console.WriteLine("11. BenchmarkDotNet: контроль обхода без GPU");
            Console.WriteLine("12. CPU-папка: файлы и свёртка последовательно (аналог GPU-папки)");
            Console.WriteLine("13. ImageSharp напрямую: загрузка в L8 без промежуточных массивов");
            string? choice = Console.ReadLine();
            if (choice is "1" or "2" or "3" or "4" or "5" or "7" or "9" or "12" or "13")
            {
                Console.WriteLine("Профиль: 0 — только время; 1 — стадии и ожидания (Enter); 2 — также подробные GPU-этапы.");
                string? profile = Console.ReadLine();
                PerformanceProfiler.Mode = string.IsNullOrWhiteSpace(profile) ? 1 : int.TryParse(profile, out int mode) && mode >= 0 && mode <= 2 ? mode : throw new ArgumentException("Неверный режим профиля");
                Console.WriteLine($"Каждый вариант: {ExperimentRunner.MeasuredRuns} измерений после отдельного прогрева.");
                Console.WriteLine($"Отчёты: {ExperimentRunner.ResultsDirectory}");
                if (PerformanceProfiler.Mode == 2)
                    Console.WriteLine("Подробный GPU-профиль добавляет синхронизацию и может менять время. Сравнивайте одинаковые режимы профилирования.");
            }
            if (choice == "10")
            {
                Console.Write("Введите путь к изображению: ");
                string? path = ReadPath();
                if (!File.Exists(path)) return;
                Console.WriteLine("byte; BoxBlur 3x3; Extend; один вычисляющий поток; блок 32x32; выделение результата включено; файлы вне замера.");
                LocalityExperiment.Run(path);
            }
            else if (choice == "11")
            {
                PerformanceProfiler.Mode = 0;
                BenchmarkRunner.Run<LocalityBenchmarks>();
            }
            else if (choice == "1")
            {
                Console.Write("Введите путь к изображению: ");
                string? path = ReadPath();
                if (!File.Exists(path)) return;
                byte[,] image = ImageIO.LoadAsGrayscaleByte(path);
                foreach (var strategy in new[] { ParallelStrategy.Sequential, ParallelStrategy.ParallelByRows, ParallelStrategy.ParallelByColumns })
                    ExperimentRunner.Run("Single-" + strategy, () =>
                    {
                        PerformanceProfiler.Set("input", Path.GetFullPath(path));
                        PerformanceProfiler.Set("timingBoundary", "Image already in RAM; convolution and result allocation only");
                        _ = ConvolutionCore.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend, strategy);
                    });
                byte[,] result = ConvolutionProcessor.Convolve(image, Kernels.BlurBox);
                string stem = Path.Combine(Path.GetDirectoryName(path)!, Path.GetFileNameWithoutExtension(path) + "_filtered");
                int index = 1;
                string output;
                do { output = stem + "_" + index++ + Path.GetExtension(path); } while (File.Exists(output));
                ImageIO.SaveImageByte(result, output);
                Console.WriteLine($"Сохранено: {output}");
            }
            else if (choice is "2" or "3" or "4" or "7" or "9" or "12" or "13")
            {
                Console.Write("Введите путь к папке: ");
                string? input = ReadPath();
                if (string.IsNullOrWhiteSpace(input) || !Directory.Exists(input)) return;
                if (choice == "12")
                    ExperimentRunner.Run("Batch-CPU-Sequential", () => BatchProcessor.ProcessImagesSequential(input,
                        OutputFolder(input, "CPU_Sequential_Output")));
                else if (choice == "2")
                    foreach (var strategy in new[] { ParallelStrategy.Sequential, ParallelStrategy.ParallelByRows, ParallelStrategy.ParallelByColumns })
                        ExperimentRunner.Run("Batch-" + strategy, () => BatchProcessor.ProcessImagesNaiveParallel(input, OutputFolder(input, "Processed_" + strategy), strategy));
                else if (choice == "3")
                {
                    int workers = ReadCount($"Вычислительных агентов (Enter для {Environment.ProcessorCount}): ", Environment.ProcessorCount);
                    int readers = ReadCount("Агентов чтения (Enter для 1): ", 1);
                    int writers = ReadCount("Агентов записи (Enter для 1): ", 1);
                    ExperimentRunner.Run($"Agents-{workers}-R{readers}-W{writers}", () => AgentProcessor.ProcessImagesWithAgents(input,
                        OutputFolder(input, "Agent_Processed_Output"), workers, readerCount: readers, writerCount: writers));
                }
                else if (choice == "4")
                    ExperimentRunner.Run("ImageSharp", () => LibraryProcessor.ProcessImagesWithImageSharp(input, OutputFolder(input, "ImageSharp_Output")));
                else if (choice == "9")
                    ExperimentRunner.Run("ImageSharp-Native", () => LibraryProcessor.ProcessImagesWithImageSharpNative(input, OutputFolder(input, "ImageSharp_Native_Output")));
                else if (choice == "13")
                    ExperimentRunner.Run("ImageSharp-Direct-L8", () => LibraryProcessor.ProcessImagesWithImageSharpDirect(input, OutputFolder(input, "ImageSharp_Direct_L8_Output")));
                else
                {
                    int cpu = ReadCount("CPU агентов (Enter для auto; 0 — без CPU-свёртки): ", Math.Max(1, Environment.ProcessorCount / 2), true);
                    int gpu = ReadCount("GPU агентов (Enter для 1; 0 — без GPU): ", 1, true);
                    int readers = ReadCount("Агентов чтения (Enter для 1): ", 1);
                    int writers = ReadCount("Агентов записи (Enter для 1): ", 1);
                    var config = new UnifiedProcessorConfig
                    {
                        CpuWorkers = cpu,
                        GpuWorkers = gpu,
                        ReaderThreads = readers,
                        WriterThreads = writers,
                        Kernel = Kernels.BlurBoxFloat,
                        Strategy = EdgeStrategy.Extend
                    };
                    ExperimentRunner.Run($"Unified-CPU{cpu}-GPU{gpu}-R{readers}-W{writers}", () =>
                    {
                        using var processor = new UnifiedProcessor(config);
                        processor.ProcessDirectory(input, OutputFolder(input, "Unified_Output"));
                    });
                }
            }
            else if (choice == "5")
            {
                Console.Write("Введите путь к файлу или папке: ");
                string? path = ReadPath();
                if (string.IsNullOrWhiteSpace(path)) return;
                if (File.Exists(path))
                {
                    byte[,] image = ImageIO.LoadAsGrayscaleByte(path);
                    using var gpu = new GpuConvolutionProcessor();
                    ExperimentRunner.Run("Single-GPU-byte", () =>
                    {
                        PerformanceProfiler.Set("input", Path.GetFullPath(path));
                        PerformanceProfiler.Set("timingBoundary", "Image in RAM; GPU already initialized; transfers and result included");
                        _ = gpu.ConvolveGpu(image, Kernels.BlurBoxFloat, EdgeStrategy.Extend);
                    });
                    string output = Path.Combine(Path.GetDirectoryName(path)!, Path.GetFileNameWithoutExtension(path) + "_gpu_processed" + Path.GetExtension(path));
                    ImageIO.SaveImageByte(gpu.ConvolveGpu(image, Kernels.BlurBoxFloat, EdgeStrategy.Extend), output);
                    Console.WriteLine($"Сохранено: {output}");
                }
                else if (Directory.Exists(path))
                    ExperimentRunner.Run("Batch-GPU-byte", () => GpuConvolutionProcessor.ProcessDirectory(path,
                        OutputFolder(path, "GPU_Batch_Output"), Kernels.BlurBoxFloat, EdgeStrategy.Extend, false));
            }
            else if (choice == "6")
            {
                string output = Path.Combine(Environment.CurrentDirectory, "Test4K");
                Directory.CreateDirectory(output);
                for (int i = 0; i < 100; i++)
                {
                    var image = new byte[2160, 3840];
                    var random = new Random(i);
                    for (int y = 0; y < 2160; y++)
                        for (int x = 0; x < 3840; x++) image[y, x] = (byte)random.Next(256);
                    ImageIO.SaveImageByte(image, Path.Combine(output, $"test_{i:000}.jpg"));
                    Console.WriteLine($"Создано {i + 1}/100");
                }
                Console.WriteLine($"Готово: {output}");
            }
            else if (choice == "8")
            {
                PerformanceProfiler.Mode = 0;
                Console.WriteLine("BenchmarkDotNet: 25 измеряемых итераций на каждый метод и размер; прогрев отдельно. Итерация может содержать несколько вызовов.");
                BenchmarkRunner.Run<ConvolutionBenchmarks>();
            }
            else Console.WriteLine("Неверный выбор.");
        }
    }


    public class ImageIO
    {
        public static byte ToByte(double value) => !(value > 0) ? (byte)0 : value >= 255 ? (byte)255 : (byte)(value + 0.5);


        public static byte[,] LoadAsGrayscaleByte(string path)
        {
            using Image<Rgba32> image = PerformanceProfiler.Measure("io.decode_read", () => Image.Load<Rgba32>(path));
            using var grayscale = PerformanceProfiler.Measure("io.grayscale_allocate");
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

            using Image<L8> image = PerformanceProfiler.Measure("io.output_allocate", () => new Image<L8>(width, height));

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

            using (PerformanceProfiler.Measure("io.encode_write")) image.Save(path);
        }

    }

    public class Kernels
    {
        // Пиксели хранятся как byte; дробные коэффициенты нужны для усреднения.
        public static readonly float[,] Sharpen =
        {
            { 0, -1, 0 },
            { -1, 5, -1 },
            { 0, -1, 0 }
        };
        public static readonly float[,] Laplacian =
        {
            { -1, -1, -1 },
            { -1, 8, -1 },
            { -1, -1, -1 }
        };
        public static readonly float[,] BlurBox =
        {
            { 1f/9f, 1f/9f, 1f/9f },
            { 1f/9f, 1f/9f, 1f/9f },
            { 1f/9f, 1f/9f, 1f/9f }
        };
        public static float[,] SharpenFloat => Sharpen;
        public static float[,] LaplacianFloat => Laplacian;
        public static float[,] BlurBoxFloat => BlurBox;
    }

    public class ConvolutionProcessor
    {
        public static byte[,] Convolve(byte[,] image, float[,] kernel, EdgeStrategy strategy = EdgeStrategy.Extend) =>
            ConvolutionCore.Convolve(image, kernel, strategy, ParallelStrategy.Sequential);
    }

    public class ParallelConvolutionProcessor
    {
        public static byte[,] ConvolveParallel(byte[,] image, float[,] kernel, EdgeStrategy strategy = EdgeStrategy.Extend) =>
            ConvolutionCore.Convolve(image, kernel, strategy, ParallelStrategy.ParallelByRows);

        public static byte[,] ConvolveParallelByColumns(byte[,] image, float[,] kernel, EdgeStrategy strategy = EdgeStrategy.Extend) =>
            ConvolutionCore.Convolve(image, kernel, strategy, ParallelStrategy.ParallelByColumns);
    }
}