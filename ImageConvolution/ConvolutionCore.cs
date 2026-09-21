using System.Numerics;
using System.Runtime.InteropServices;

namespace ImageConvolution;

internal static class ConvolutionCore
{
    public static T[,] Convolve<T>(T[,] image, T[,] kernel, EdgeStrategy edge, ParallelStrategy parallel)
        where T : unmanaged, IFloatingPointIeee754<T>
    {
        ConvolutionValidation.Validate(image, kernel, edge);
        if (!Enum.IsDefined(parallel)) throw new ArgumentOutOfRangeException(nameof(parallel));
        int height = image.GetLength(0), width = image.GetLength(1);
        int kh = kernel.GetLength(0), kw = kernel.GetLength(1);
        int oy = kh / 2, ox = kw / 2;
        var result = new T[height, width];
        var weights = new T[kernel.Length];
        var offsets = new int[kernel.Length];
        bool hasInterior = height >= kh && width >= kw;
        for (int ky = 0, i = 0; ky < kh; ky++)
            for (int kx = 0; kx < kw; kx++, i++)
            {
                weights[i] = kernel[ky, kx];
                offsets[i] = hasInterior ? (oy - ky) * width + ox - kx : 0;
            }
        void Line(int line, bool columns)
        {
            var source = MemoryMarshal.CreateReadOnlySpan(ref image[0, 0], image.Length);
            var target = MemoryMarshal.CreateSpan(ref result[0, 0], result.Length);
            int count = columns ? height : width;
            for (int position = 0; position < count; position++)
            {
                int y = columns ? position : line, x = columns ? line : position;
                int center = y * width + x;
                T sum = T.Zero;
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
                            T value = T.Zero;
                            if (edge == EdgeStrategy.Extend)
                                value = source[Math.Clamp(sy, 0, height - 1) * width + Math.Clamp(sx, 0, width - 1)];
                            else if ((uint)sy < (uint)height && (uint)sx < (uint)width)
                                value = source[sy * width + sx];
                            sum += value * weights[i];
                        }
                }
                target[center] = sum;
            }
        }
        if (parallel == ParallelStrategy.ParallelByRows)
            Parallel.For(0, height, y => Line(y, false));
        else if (parallel == ParallelStrategy.ParallelByColumns)
            Parallel.For(0, width, x => Line(x, true));
        else
            for (int y = 0; y < height; y++) Line(y, false);
        return result;
    }
}
