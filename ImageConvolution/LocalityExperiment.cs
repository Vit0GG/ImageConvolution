using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.Globalization;
using System.Text;

using BenchmarkDotNet.Attributes;

namespace ImageConvolution;

public enum SequentialTraversal { Rows, Columns, BlockedColumns }

public static class LocalityExperiment
{
    public static byte[,] Convolve(byte[,] image, float[,] kernel, EdgeStrategy edge,
        SequentialTraversal traversal, int blockSize = 32) =>
        ConvolutionCore.Convolve(image, kernel, edge, ParallelStrategy.Sequential, traversal, blockSize);

    public static void Run(string path)
    {
        var image = ImageIO.LoadAsGrayscaleByte(path);
        PerformanceProfiler.Mode = 0;
        foreach (var traversal in Enum.GetValues<SequentialTraversal>())
            ExperimentRunner.Run("Locality-" + traversal, () =>
            {
                var result = Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend, traversal);
                GC.KeepAlive(result);
            });
    }

    public static void ParallelControl(string path, int partitions, int measuredRuns = 7)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(partitions);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(measuredRuns);
        PerformanceProfiler.Mode = 0;
        var image = ImageIO.LoadAsGrayscaleByte(path);
        var variants = new (string Name, ParallelStrategy Strategy, int Partitions, SequentialTraversal Traversal)[]
        {
            ("OriginalRows", ParallelStrategy.ParallelByRows, 0, SequentialTraversal.Rows),
            ("OriginalColumns", ParallelStrategy.ParallelByColumns, 0, SequentialTraversal.Rows),
            ("EqualPartitionsRows", ParallelStrategy.ParallelByRows, partitions, SequentialTraversal.Rows),
            ("EqualPartitionsColumns", ParallelStrategy.ParallelByColumns, partitions, SequentialTraversal.Rows),
            ("EqualPartitionsTiledColumns32", ParallelStrategy.ParallelByColumns, partitions, SequentialTraversal.BlockedColumns)
        };
        var expected = ConvolutionCore.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend, ParallelStrategy.Sequential);
        foreach (var variant in variants)
        {
            var actual = ConvolutionCore.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend,
                variant.Strategy, variant.Traversal, partitionCount: variant.Partitions);
            for (int y = 0; y < expected.GetLength(0); y++)
                for (int x = 0; x < expected.GetLength(1); x++)
                    if (expected[y, x] != actual[y, x])
                        throw new InvalidOperationException($"Различие результата {variant.Name} в ({y}, {x})");
            GC.KeepAlive(actual);
        }
        var samples = variants.ToDictionary(v => v.Name, _ => new List<double>(measuredRuns));
        var rows = new StringBuilder("run,variant,elapsed_ms\n");
        for (int run = 0; run < measuredRuns + 1; run++)
        {
            for (int offset = 0; offset < variants.Length; offset++)
            {
                var variant = variants[(offset + run) % variants.Length];
                var timer = Stopwatch.StartNew();
                var result = ConvolutionCore.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend,
                    variant.Strategy, variant.Traversal, partitionCount: variant.Partitions);
                timer.Stop();
                GC.KeepAlive(result);
                if (run == 0) continue;
                double elapsed = timer.Elapsed.TotalMilliseconds;
                samples[variant.Name].Add(elapsed);
                rows.AppendLine(FormattableString.Invariant($"{run},{variant.Name},{elapsed:F6}"));
            }
        }
        string folder = Path.Combine(Environment.CurrentDirectory, "MeasurementResults",
            "LocalityParallelControl_" + DateTime.Now.ToString("yyyyMMdd_HHmmss") + "_" + Guid.NewGuid().ToString("N")[..8]);
        Directory.CreateDirectory(folder);
        File.WriteAllText(Path.Combine(folder, "times.csv"), rows.ToString());
        var report = new StringBuilder();
        report.AppendLine($"Один файл в памяти: {image.GetLength(1)}x{image.GetLength(0)}, byte, BoxBlur 3x3, Extend");
        report.AppendLine($"Все выходы побитно совпадают; прогрев=1; измерений={measuredRuns}; фиксированных разделов={partitions}; профиль=0");
        report.AppendLine("Разделы задают число диапазонов строк или столбцов, а не гарантированное число одновременно работающих потоков.");
        foreach (var variant in variants)
        {
            var values = samples[variant.Name].Order().ToArray();
            double median = values.Length % 2 == 1 ? values[values.Length / 2]
                : (values[values.Length / 2 - 1] + values[values.Length / 2]) / 2;
            report.AppendLine(FormattableString.Invariant($"{variant.Name}: медиана={median:F3} мс; минимум={values[0]:F3}; максимум={values[^1]:F3}"));
        }
        report.AppendLine("Замер включает создание массива результата и свёртку; чтение и запись файла исключены.");
        report.AppendLine("Это контроль порядка обхода и планирования; без аппаратных счётчиков он не измеряет промахи кэша.");
        File.WriteAllText(Path.Combine(folder, "summary.txt"), report.ToString());
        Console.WriteLine(report);
        Console.WriteLine($"Результаты: {folder}");
    }

    public static void HardwareWorkload(string[] args)
    {
        bool parallelRows = args.Length > 1 && args[1] == "ParallelRows";
        bool parallelColumns = args.Length > 1 && args[1] == "ParallelColumns";
        SequentialTraversal traversal = SequentialTraversal.Rows;
        if (args.Length is < 4 or > 5
            || (!(parallelRows || parallelColumns) && (!Enum.TryParse(args[1], true, out traversal) || !Enum.IsDefined(traversal)))
            || !int.TryParse(args[3], out int iterations) || iterations < 1)
            throw new ArgumentException("--locality-profile Rows|Columns|ParallelRows|ParallelColumns путь_к_файлу число_повторов [отчёт.json]");
        PerformanceProfiler.Mode = 0;
        var image = ImageIO.LoadAsGrayscaleByte(args[2]);
        byte[,] Apply() => parallelRows || parallelColumns
            ? ConvolutionCore.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend,
                parallelRows ? ParallelStrategy.ParallelByRows : ParallelStrategy.ParallelByColumns)
            : Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend, traversal);
        for (int i = 0; i < 5; i++) GC.KeepAlive(Apply());
        double checksum = 0;
        DateTime startedUtc = DateTime.UtcNow;
        var timer = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            var result = Apply();
            checksum += result[i % result.GetLength(0), i % result.GetLength(1)];
        }
        timer.Stop();
        DateTime endedUtc = DateTime.UtcNow;
        if (args.Length == 5)
            File.WriteAllText(args[4], System.Text.Json.JsonSerializer.Serialize(new
            {
                variant = args[1],
                processId = Environment.ProcessId,
                startedUtc,
                endedUtc,
                iterations,
                width = image.GetLength(1),
                height = image.GetLength(0),
                elapsedMilliseconds = timer.Elapsed.TotalMilliseconds,
                checksum,
                scope = "Convolution including result allocation; excludes decoding and five warmups"
            }, new System.Text.Json.JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine(FormattableString.Invariant($"{args[1]}; iterations={iterations}; elapsed_ms={timer.Elapsed.TotalMilliseconds:F3}; checksum={checksum:R}"));
    }
}

[MemoryDiagnoser]
[IterationCount(ExperimentRunner.MeasuredRuns)]
[WarmupCount(3)]
[Outliers(Perfolizer.Mathematics.OutlierDetection.OutlierMode.DontRemove)]
[ExcludeFromCodeCoverage]
public class LocalityBenchmarks
{
    [Params(1024, 3840)]
    public int Width { get; set; }
    private byte[,] image = null!;

    [GlobalSetup]
    public void Setup()
    {
        PerformanceProfiler.Mode = 0;
        image = new byte[Width == 3840 ? 2160 : 1024, Width];
        var random = new Random(42);
        for (int y = 0; y < image.GetLength(0); y++)
            for (int x = 0; x < Width; x++) image[y, x] = (byte)random.Next(256);
    }

    [Benchmark(Baseline = true)]
    public byte[,] Rows() => LocalityExperiment.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend, SequentialTraversal.Rows);

    [Benchmark]
    public byte[,] Columns() => LocalityExperiment.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend, SequentialTraversal.Columns);

    [Benchmark]
    public byte[,] BlockedColumns() => LocalityExperiment.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend, SequentialTraversal.BlockedColumns);
}