using System.Runtime.InteropServices;

namespace ImageConvolution;

internal static class ConvolutionCore
{
    public static byte[,] Convolve(byte[,] image, float[,] kernel, EdgeStrategy edge, ParallelStrategy parallel,
        SequentialTraversal traversal = SequentialTraversal.Rows, int blockSize = 32, int partitionCount = 0)
    {
        ConvolutionValidation.Validate(image, kernel, edge);
        if (!Enum.IsDefined(parallel)) throw new ArgumentOutOfRangeException(nameof(parallel));
        if (!Enum.IsDefined(traversal)) throw new ArgumentOutOfRangeException(nameof(traversal));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(blockSize);
        ArgumentOutOfRangeException.ThrowIfNegative(partitionCount);
        if (partitionCount > 0 && parallel == ParallelStrategy.Sequential)
            throw new ArgumentException("Контрольные разделы требуют параллельного выполнения", nameof(partitionCount));
        if (parallel != ParallelStrategy.Sequential && traversal != SequentialTraversal.Rows
            && !(partitionCount > 0 && parallel == ParallelStrategy.ParallelByColumns && traversal == SequentialTraversal.BlockedColumns))
            throw new ArgumentException("Контрольный порядок обхода требует последовательного выполнения", nameof(traversal));
        using var measurement = PerformanceProfiler.Measure(PerformanceProfiler.IsActive ? "cpu.convolve." + parallel : "");
        PerformanceProfiler.DescribeArray(image);
        if (PerformanceProfiler.IsActive)
            PerformanceProfiler.Set("kernel", new { height = kernel.GetLength(0), width = kernel.GetLength(1), edge = edge.ToString() });
        int height = image.GetLength(0), width = image.GetLength(1);
        int kh = kernel.GetLength(0), kw = kernel.GetLength(1);
        int oy = kh / 2, ox = kw / 2;
        var result = new byte[height, width];
        var weights = new float[kernel.Length];
        var offsets = new int[kernel.Length];
        bool hasInterior = height >= kh && width >= kw;
        for (int ky = 0, i = 0; ky < kh; ky++)
            for (int kx = 0; kx < kw; kx++, i++)
            {
                weights[i] = kernel[ky, kx];
                offsets[i] = hasInterior ? (oy - ky) * width + ox - kx : 0;
            }
        void Line(int line, bool columns, int start = 0, int end = int.MaxValue)
        {
            var source = MemoryMarshal.CreateReadOnlySpan(ref image[0, 0], image.Length);
            var target = MemoryMarshal.CreateSpan(ref result[0, 0], result.Length);
            int count = Math.Min(columns ? height : width, end);
            for (int position = start; position < count; position++)
            {
                int y = columns ? position : line, x = columns ? line : position;
                int center = y * width + x;
                float sum = 0;
                if (hasInterior && y >= oy && y < height - oy && x >= ox && x < width - ox)
                {
                    for (int i = 0; i < weights.Length; i++) sum += source[center + offsets[i]] * weights[i];
                }
                else
                {
                    for (int ky = 0, i = 0; ky < kh; ky++)
                        for (int kx = 0; kx < kw; kx++, i++)
                        {
                            int sy = y + oy - ky, sx = x + ox - kx;
                            byte value = 0;
                            if (edge == EdgeStrategy.Extend)
                                value = source[Math.Clamp(sy, 0, height - 1) * width + Math.Clamp(sx, 0, width - 1)];
                            else if ((uint)sy < (uint)height && (uint)sx < (uint)width)
                                value = source[sy * width + sx];
                            sum += value * weights[i];
                        }
                }
                target[center] = ImageIO.ToByte(sum);
            }
        }
        if (partitionCount > 0)
        {
            bool columns = parallel == ParallelStrategy.ParallelByColumns;
            int lines = columns ? width : height;
            Parallel.For(0, partitionCount, new ParallelOptions { MaxDegreeOfParallelism = partitionCount }, partition =>
            {
                int first = (int)((long)partition * lines / partitionCount);
                int last = (int)((long)(partition + 1) * lines / partitionCount);
                if (columns && traversal == SequentialTraversal.BlockedColumns)
                {
                    for (int y0 = 0; y0 < height; y0 += blockSize)
                    {
                        int y1 = Math.Min(y0 + blockSize, height);
                        for (int x0 = first; x0 < last; x0 += blockSize)
                        {
                            int x1 = Math.Min(x0 + blockSize, last);
                            for (int x = x0; x < x1; x++) Line(x, true, y0, y1);
                        }
                    }
                }
                else
                    for (int line = first; line < last; line++) Line(line, columns);
            });
        }
        else if (parallel == ParallelStrategy.ParallelByRows)
            Parallel.For(0, height, y => Line(y, false));
        else if (parallel == ParallelStrategy.ParallelByColumns)
            Parallel.For(0, width, x => Line(x, true));
        else if (traversal == SequentialTraversal.Columns)
            for (int x = 0; x < width; x++) Line(x, true);
        else if (traversal == SequentialTraversal.BlockedColumns)
        {
            for (int y0 = 0; y0 < height;)
            {
                int y1 = y0 + Math.Min(blockSize, height - y0);
                for (int x0 = 0; x0 < width;)
                {
                    int x1 = x0 + Math.Min(blockSize, width - x0);
                    for (int x = x0; x < x1; x++) Line(x, true, y0, y1);
                    x0 = x1;
                }
                y0 = y1;
            }
        }
        else
            for (int y = 0; y < height; y++) Line(y, false);
        return result;
    }
}