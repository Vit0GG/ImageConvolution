namespace ImageConvolution;

internal static class ConvolutionValidation
{
    public static void Validate<T, TKernel>(T[,] image, TKernel[,] kernel, EdgeStrategy strategy)
        where TKernel : System.Numerics.IFloatingPointIeee754<TKernel>
    {
        ArgumentNullException.ThrowIfNull(image);
        if (image.GetLength(0) == 0 || image.GetLength(1) == 0)
            throw new ArgumentException("Изображение не должно быть пустым", nameof(image));
        ValidateKernel(kernel, strategy);
    }

    public static void ValidateKernel<T>(T[,] kernel, EdgeStrategy strategy)
        where T : System.Numerics.IFloatingPointIeee754<T>
    {
        ArgumentNullException.ThrowIfNull(kernel);
        if (kernel.GetLength(0) % 2 != 1 || kernel.GetLength(1) % 2 != 1)
            throw new ArgumentException("Размеры ядра должны быть положительными и нечётными", nameof(kernel));
        foreach (T value in kernel)
            if (!T.IsFinite(value)) throw new ArgumentException("Коэффициенты ядра должны быть конечными", nameof(kernel));
        if (!Enum.IsDefined(strategy)) throw new ArgumentOutOfRangeException(nameof(strategy));
    }
}

internal static class ImageFiles
{
    private static readonly HashSet<string> Extensions = new(StringComparer.OrdinalIgnoreCase)
    {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
    };

    public static string[] GetFiles(string directory) => Directory.EnumerateFiles(directory)
        .Where(f => Extensions.Contains(Path.GetExtension(f))).OrderBy(f => f, StringComparer.Ordinal).ToArray();

    public static int GetInFlightLimit(string[] files, long budget, int bytesPerPixel)
    {
        if (budget == 0) budget = Math.Clamp(GC.GetGCMemoryInfo().TotalAvailableMemoryBytes / 8, 256L * 1024 * 1024, 2L * 1024 * 1024 * 1024);
        long largest = 1;
        foreach (var file in files)
        {
            var info = SixLabors.ImageSharp.Image.Identify(file);
            largest = Math.Max(largest, checked((long)info.Width * info.Height * bytesPerPixel));
        }
        return (int)Math.Clamp(budget / largest, 1, Math.Max(1, files.Length));
    }

    public static void ValidateDirectories(string input, string output)
    {
        if (string.Equals(Path.TrimEndingDirectorySeparator(Path.GetFullPath(input)),
            Path.TrimEndingDirectorySeparator(Path.GetFullPath(output)), StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException("Исходная и выходная папки должны различаться", nameof(output));
    }
}
