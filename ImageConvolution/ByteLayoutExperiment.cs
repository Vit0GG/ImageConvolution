using System.Diagnostics;
using System.Text;
using System.Text.Json;

namespace ImageConvolution;

// A controlled experiment. Padding is outside the logical image and is never convolved.
public static class ByteLayoutExperiment
{
    private readonly record struct Variant(string Name, bool ColumnStorage, bool ColumnTraversal, int Padding);

    private static readonly Variant[] Variants =
    [
        new("Rows-P0", false, false, 0),
        new("Columns-P0", false, true, 0),
        new("Columns-P1", false, true, 1),
        new("Columns-P3", false, true, 3),
        new("Columns-P8", false, true, 8),
        new("Columns-P16", false, true, 16),
        new("Columns-P64", false, true, 64),
        new("Columns-P128", false, true, 128),
        new("ColumnStorage-Columns", true, true, 0),
        new("ColumnStorage-Rows", true, false, 0)
    ];

    public static void Run(string[] args)
    {
        if (args.Length < 3 || args.Length > 6)
            throw new ArgumentException("--byte-layout путь_к_изображению папка_отчёта [число_измерений] | --byte-layout путь папка --hardware вариант [число_повторов]");
        string path = Path.GetFullPath(args[1]);
        string reportDirectory = Path.GetFullPath(args[2]);
        Directory.CreateDirectory(reportDirectory);
        var image = ImageIO.LoadAsGrayscaleByte(path);
        int height = image.GetLength(0), width = image.GetLength(1);
        var inputs = new Dictionary<int, byte[]>();
        foreach (int padding in Variants.Where(v => !v.ColumnStorage).Select(v => v.Padding).Distinct())
        {
            var data = new byte[checked(height * (width + padding))];
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++) data[y * (width + padding) + x] = image[y, x];
            inputs.Add(padding, data);
        }
        var columnInput = new byte[checked(width * height)];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) columnInput[x * height + y] = image[y, x];

        byte[] Apply(Variant variant) => Convolve(variant.ColumnStorage ? columnInput : inputs[variant.Padding],
            width, height, variant.ColumnStorage, variant.ColumnTraversal, variant.Padding);

        var expected = ConvolutionProcessor.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend);
        void Check(Variant variant)
        {
            byte[] actual = Apply(variant);
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    int index = variant.ColumnStorage ? x * height + y : y * (width + variant.Padding) + x;
                    if (actual[index] != expected[y, x])
                        throw new InvalidOperationException($"{variant.Name}: результат отличается в ({x},{y})");
                }
        }

        if (args.Length >= 4 && args[3] == "--hardware")
        {
            if (args.Length < 5) throw new ArgumentException("Нужно указать вариант для аппаратного профиля");
            Variant variant = Variants.Single(v => v.Name == args[4]);
            int iterations = args.Length == 6 ? int.Parse(args[5]) : 30;
            if (iterations < 1) throw new ArgumentOutOfRangeException(nameof(iterations));
            Check(variant);
            for (int i = 0; i < 5; i++) GC.KeepAlive(Apply(variant));
            DateTime startedUtc = DateTime.UtcNow;
            var timer = Stopwatch.StartNew();
            long checksum = 0;
            for (int i = 0; i < iterations; i++)
            {
                byte[] result = Apply(variant);
                checksum += result[variant.ColumnStorage ? (i % width) * height + i % height : (i % height) * (width + variant.Padding) + i % width];
            }
            timer.Stop();
            File.WriteAllText(Path.Combine(reportDirectory, "workload.json"), JsonSerializer.Serialize(new
            {
                variant = variant.Name,
                processId = Environment.ProcessId,
                startedUtc,
                endedUtc = DateTime.UtcNow,
                iterations,
                width,
                height,
                pixelType = "byte",
                inputPadding = variant.Padding,
                outputPadding = variant.Padding,
                elapsedMilliseconds = timer.Elapsed.TotalMilliseconds,
                checksum,
                scope = "Convolution and result allocation; JPEG decode and five warmups excluded"
            }, new JsonSerializerOptions { WriteIndented = true }));
            return;
        }

        int measuredRuns = args.Length == 4 ? int.Parse(args[3]) : 25;
        if (measuredRuns < 1) throw new ArgumentOutOfRangeException(nameof(measuredRuns));
        foreach (Variant variant in Variants) Check(variant);
        var samples = Variants.ToDictionary(v => v.Name, _ => new List<double>(measuredRuns));
        var csv = new StringBuilder("run,variant,elapsed_ms\n");
        const int warmups = 5;
        for (int run = -warmups; run < measuredRuns; run++)
        {
            for (int offset = 0; offset < Variants.Length; offset++)
            {
                Variant variant = Variants[(offset + run + Variants.Length) % Variants.Length];
                var timer = Stopwatch.StartNew();
                byte[] result = Apply(variant);
                timer.Stop();
                GC.KeepAlive(result);
                if (run < 0) continue;
                double ms = timer.Elapsed.TotalMilliseconds;
                samples[variant.Name].Add(ms);
                csv.AppendLine(FormattableString.Invariant($"{run + 1},{variant.Name},{ms:F6}"));
            }
            if (run >= 0 && (run + 1) % 5 == 0) Console.WriteLine($"Byte layout: {run + 1}/{measuredRuns}");
        }
        File.WriteAllText(Path.Combine(reportDirectory, "times.csv"), csv.ToString());
        var summary = new StringBuilder();
        summary.AppendLine($"{width}x{height}; byte pixels; BoxBlur 3x3; Extend; {warmups} warmups; {measuredRuns} measurements per variant.");
        summary.AppendLine("Same logical pixels and convolution; every variant matched the main CPU implementation bit for bit.");
        summary.AppendLine("Padding is unused space after each input/output row. Timing includes result allocation and convolution, excludes JPEG I/O.");
        summary.AppendLine("Padding of 64 byte pixels equals the old 8-double (64-byte) padding in physical size.");
        foreach (Variant variant in Variants)
        {
            double[] sorted = samples[variant.Name].Order().ToArray();
            double median = sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] :
                (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
            summary.AppendLine(FormattableString.Invariant($"{variant.Name}: median={median:F3} ms; min={sorted[0]:F3}; max={sorted[^1]:F3}"));
        }
        File.WriteAllText(Path.Combine(reportDirectory, "summary.txt"), summary.ToString());
        Console.WriteLine(summary);
    }

    private static byte[] Convolve(byte[] source, int width, int height, bool columnStorage, bool columnTraversal, int padding)
    {
        int strideY = columnStorage ? 1 : width + padding;
        int strideX = columnStorage ? height : 1;
        var target = new byte[source.Length];
        var weights = new float[9];
        var offsets = new int[9];
        for (int ky = 0, i = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++, i++)
            {
                weights[i] = Kernels.BlurBox[ky, kx];
                offsets[i] = (1 - ky) * strideY + (1 - kx) * strideX;
            }
        void Line(int line)
        {
            int length = columnTraversal ? height : width;
            for (int position = 0; position < length; position++)
            {
                int y = columnTraversal ? position : line;
                int x = columnTraversal ? line : position;
                int center = y * strideY + x * strideX;
                float sum = 0;
                if (y > 0 && y < height - 1 && x > 0 && x < width - 1)
                    for (int i = 0; i < 9; i++) sum += source[center + offsets[i]] * weights[i];
                else
                    for (int ky = 0, i = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++, i++)
                        {
                            int sy = Math.Clamp(y + 1 - ky, 0, height - 1);
                            int sx = Math.Clamp(x + 1 - kx, 0, width - 1);
                            sum += source[sy * strideY + sx * strideX] * weights[i];
                        }
                target[center] = ImageIO.ToByte(sum);
            }
        }
        if (columnTraversal)
            Parallel.For(0, width, Line);
        else
            Parallel.For(0, height, Line);
        return target;
    }
}