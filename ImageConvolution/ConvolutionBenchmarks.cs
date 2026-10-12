using System;
using System.Diagnostics.CodeAnalysis;

using BenchmarkDotNet.Attributes;

namespace ImageConvolution
{
    [MemoryDiagnoser]
    [IterationCount(ExperimentRunner.MeasuredRuns)]
    [WarmupCount(3)]
    [Outliers(Perfolizer.Mathematics.OutlierDetection.OutlierMode.DontRemove)]
    [ExcludeFromCodeCoverage]
    public class ConvolutionBenchmarks
    {
        private byte[,] image = null!;
        private GpuConvolutionProcessor gpuProcessor = null!;

        [Params(512, 1024)]
        public int ImageSize { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            image = new byte[ImageSize, ImageSize];

            var rand = new Random(42);
            for (int y = 0; y < ImageSize; y++)
            {
                for (int x = 0; x < ImageSize; x++)
                {
                    image[y, x] = (byte)rand.Next(0, 256);
                }
            }

            gpuProcessor = new GpuConvolutionProcessor();

            _ = gpuProcessor.ConvolveGpu(image, Kernels.BlurBox, EdgeStrategy.Extend);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            gpuProcessor?.Dispose();
        }

        [Benchmark(Baseline = true, Description = "1. CPU (Один поток)")]
        public byte[,] CpuSequential()
        {
            return ConvolutionProcessor.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend);
        }

        [Benchmark(Description = "2. CPU (Parallel.For)")]
        public byte[,] CpuParallel()
        {
            return ParallelConvolutionProcessor.ConvolveParallel(image, Kernels.BlurBox, EdgeStrategy.Extend);
        }

        [Benchmark(Description = "CPU (Parallel.For по столбцам)")]
        public byte[,] CpuParallelColumns() => ParallelConvolutionProcessor.ConvolveParallelByColumns(image, Kernels.BlurBox);

        [Benchmark(Description = "ImageSharp BoxBlur 3x3 + преобразования")]
        public byte[,] ImageSharpBox() => LibraryProcessor.ConvolveBox(image);

        [Benchmark(Description = "3. GPU (PCIe + Math)")]
        public byte[,] GpuFullCycle()
        {
            return gpuProcessor.ConvolveGpu(image, Kernels.BlurBox, EdgeStrategy.Extend);
        }

        [Benchmark(Description = "4. GPU (запуск + ядро + синхронизация)")]
        public void GpuMathOnly()
        {
            gpuProcessor.ConvolveGpuMathOnly_ForBenchmark(Kernels.BlurBox, EdgeStrategy.Extend);
        }
    }
}