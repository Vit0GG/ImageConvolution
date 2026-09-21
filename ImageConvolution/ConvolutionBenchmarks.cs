using System;
using BenchmarkDotNet.Attributes;

namespace ImageConvolution
{
    [MemoryDiagnoser]
    public class ConvolutionBenchmarks
    {        
        private double[,] imageDouble = null!;
        private float[,] imageFloat = null!;
        private GpuConvolutionProcessor gpuProcessor = null!;
        
        [Params(512, 1024)]
        public int ImageSize { get; set; }

        [GlobalSetup]
        public void Setup()
        {
            imageDouble = new double[ImageSize, ImageSize];
            imageFloat = new float[ImageSize, ImageSize];
            
            var rand = new Random(42);
            for (int y = 0; y < ImageSize; y++)
            {
                for (int x = 0; x < ImageSize; x++)
                {
                    double val = rand.Next(0, 256);
                    imageDouble[y, x] = val;
                    imageFloat[y, x] = (float)val;
                }
            }

            gpuProcessor = new GpuConvolutionProcessor();

            _ = gpuProcessor.ConvolveGpu(imageFloat, Kernels.BlurBoxFloat, EdgeStrategy.Extend);
        }

        [GlobalCleanup]
        public void Cleanup()
        {
            gpuProcessor?.Dispose();
        }

        [Benchmark(Baseline = true, Description = "1. CPU (Один поток)")]
        public double[,] CpuSequential()
        {
            return ConvolutionProcessor.Convolve(imageDouble, Kernels.BlurBox, EdgeStrategy.Extend);
        }

        [Benchmark(Description = "2. CPU (Parallel.For)")]
        public double[,] CpuParallel()
        {
            return ParallelConvolutionProcessor.ConvolveParallel(imageDouble, Kernels.BlurBox, EdgeStrategy.Extend);
        }

        [Benchmark(Description = "CPU (Parallel.For по столбцам)")]
        public double[,] CpuParallelColumns() => ParallelConvolutionProcessor.ConvolveParallelByColumns(imageDouble, Kernels.BlurBox);

        [Benchmark(Description = "CPU (float, один поток)")]
        public float[,] CpuFloat() => UnifiedProcessor.ConvolveCpu(imageFloat, Kernels.BlurBoxFloat, EdgeStrategy.Extend);

        [Benchmark(Description = "ImageSharp BoxBlur 3x3 + преобразования")]
        public float[,] ImageSharpBox() => LibraryProcessor.ConvolveBox(imageFloat);

        [Benchmark(Description = "3. GPU (PCIe + Math)")]
        public float[,] GpuFullCycle()
        {
            return gpuProcessor.ConvolveGpu(imageFloat, Kernels.BlurBoxFloat, EdgeStrategy.Extend);
        }

        [Benchmark(Description = "4. GPU (запуск + ядро + синхронизация)")]
        public void GpuMathOnly()
        {
            gpuProcessor.ConvolveGpuMathOnly_ForBenchmark(Kernels.BlurBoxFloat, EdgeStrategy.Extend);
        }
    }
}
