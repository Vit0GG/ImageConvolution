using System;
using System.Diagnostics;
using System.IO;
using System.Linq;

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Runtime.Cuda;

namespace ImageConvolution
{
    public class GpuConvolutionProcessor : IDisposable
    {
        private readonly Context context;
        public readonly Accelerator accelerator;
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, int> convolutionKernel;

        public GpuConvolutionProcessor()
        {
            context = Context.Create(builder => builder.Cuda().Optimize(OptimizationLevel.O2));

            var gpuDevice = context.GetPreferredDevice(false);
            if (gpuDevice != null && gpuDevice.AcceleratorType == AcceleratorType.Cuda)
            {
                accelerator = gpuDevice.CreateAccelerator(context);
            }
            else
            {
                Console.WriteLine("Предупреждение: CUDA-совместимый GPU не найден. Попытка использовать эмулятор CPU.");
                var cpuDevice = context.Devices.FirstOrDefault(d => d.AcceleratorType == AcceleratorType.CPU);
                if (cpuDevice != null)
                {
                    accelerator = cpuDevice.CreateAccelerator(context);
                }
                else
                {
                    throw new InvalidOperationException("Не найден ни CUDA GPU, ни CPU Accelerator.");
                }
            }

            Console.WriteLine($"Используется устройство: {accelerator.Name}");
            convolutionKernel = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, ArrayView2D<float, Stride2D.DenseX>, int, int, int>(ConvolutionKernel);
        }

        static void ConvolutionKernel(
            Index2D index,
            ArrayView2D<float, Stride2D.DenseX> input,
            ArrayView2D<float, Stride2D.DenseX> output,
            ArrayView2D<float, Stride2D.DenseX> kernel,
            int strategy,
            int offsetX,
            int offsetY)
        {
            int x = index.X;
            int y = index.Y;

            int imgHeight = input.IntExtent.Y;
            int imgWidth = input.IntExtent.X;
            int kHeight = kernel.IntExtent.Y;
            int kWidth = kernel.IntExtent.X;

            float sum = 0.0f;

            for (int ky = 0; ky < kHeight; ky++)
            {
                for (int kx = 0; kx < kWidth; kx++)
                {
                    int pixelY = y + ky - offsetY;
                    int pixelX = x + kx - offsetX;

                    float val = 0.0f;

                    if (strategy == 0)
                    {
                        int clampedY = pixelY < 0 ? 0 : (pixelY >= imgHeight ? imgHeight - 1 : pixelY);
                        int clampedX = pixelX < 0 ? 0 : (pixelX >= imgWidth ? imgWidth - 1 : pixelX);
                        val = input[clampedY, clampedX];
                    }
                    else
                    {
                        if (pixelY >= 0 && pixelY < imgHeight && pixelX >= 0 && pixelX < imgWidth)
                        {
                            val = input[pixelY, pixelX];
                        }
                    }
                    sum += val * kernel[ky, kx];
                }
            }
            output[y, x] = sum;
        }

        public float[,] ConvolveGpu(float[,] image, float[,] kernelData, EdgeStrategy strategy)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);

            using var sourceBuffer = accelerator.Allocate2DDenseX<float>(new LongIndex2D(height, width));
            using var targetBuffer = accelerator.Allocate2DDenseX<float>(new LongIndex2D(height, width));
            using var kernelBuffer = accelerator.Allocate2DDenseX<float>(new LongIndex2D(kernelData.GetLength(0), kernelData.GetLength(1)));

            sourceBuffer.CopyFromCPU(image);
            kernelBuffer.CopyFromCPU(kernelData);

            int offsetX = kernelData.GetLength(1) / 2;
            int offsetY = kernelData.GetLength(0) / 2;

            convolutionKernel(new Index2D(width, height), sourceBuffer.View, targetBuffer.View, kernelBuffer.View, (int)strategy, offsetX, offsetY);

            return targetBuffer.GetAsArray2D();
        }

        public static void ProcessDirectory(string inputDirectory, string outputDirectory, float[,] kernel, EdgeStrategy strategy)
        {
            if (!Directory.Exists(inputDirectory))
            {
                Console.WriteLine("Ошибка: Указанная папка не существует.");
                return;
            }
            Directory.CreateDirectory(outputDirectory);

            string[] files = Directory.GetFiles(inputDirectory, "*.jpg").OrderBy(f => f).ToArray();
            if (files.Length == 0)
            {
                Console.WriteLine("Файлы .jpg не найдены.");
                return;
            }

            Console.WriteLine($"Найдено файлов для обработки на GPU: {files.Length}");

            using var gpuProcessor = new GpuConvolutionProcessor();

            float[,] firstImage = ImageIO.LoadAsGrayscaleFloat(files[0]);
            int height = firstImage.GetLength(0);
            int width = firstImage.GetLength(1);

            using var sourceBuffer = gpuProcessor.accelerator.Allocate2DDenseX<float>(new LongIndex2D(height, width));
            using var targetBuffer = gpuProcessor.accelerator.Allocate2DDenseX<float>(new LongIndex2D(height, width));
            using var kernelBuffer = gpuProcessor.accelerator.Allocate2DDenseX<float>(new LongIndex2D(kernel.GetLength(0), kernel.GetLength(1)));

            kernelBuffer.CopyFromCPU(kernel);
            int offsetX = kernel.GetLength(1) / 2;
            int offsetY = kernel.GetLength(0) / 2;

            var totalStopwatch = Stopwatch.StartNew();
            long fullGpuPipelineTimeMs = 0;
            long computeOnlyTimeMs = 0;

            for (int i = 0; i < files.Length; i++)
            {
                float[,] imageData;
                if (i == 0)
                {
                    imageData = firstImage;
                }
                else
                {
                    imageData = ImageIO.LoadAsGrayscaleFloat(files[i]);
                }

                if (imageData.GetLength(0) != height || imageData.GetLength(1) != width)
                {
                    Console.WriteLine($"\nПропуск файла {Path.GetFileName(files[i])} из-за другого размера.");
                    continue;
                }

                var fullPipelineStopwatch = Stopwatch.StartNew();
                sourceBuffer.CopyFromCPU(imageData);
                gpuProcessor.convolutionKernel(new Index2D(width, height), sourceBuffer.View, targetBuffer.View, kernelBuffer.View, (int)strategy, offsetX, offsetY);
                var resultData = targetBuffer.GetAsArray2D();
                fullPipelineStopwatch.Stop();
                fullGpuPipelineTimeMs += fullPipelineStopwatch.ElapsedMilliseconds;

                var computeOnlyStopwatch = Stopwatch.StartNew();
                sourceBuffer.CopyFromCPU(imageData);
                gpuProcessor.convolutionKernel(new Index2D(width, height), sourceBuffer.View, targetBuffer.View, kernelBuffer.View, (int)strategy, offsetX, offsetY);
                gpuProcessor.accelerator.Synchronize();
                computeOnlyStopwatch.Stop();
                computeOnlyTimeMs += computeOnlyStopwatch.ElapsedMilliseconds;

                string fileName = Path.GetFileName(files[i]);
                string savePath = Path.Combine(outputDirectory, fileName);
                ImageIO.SaveImageFloat(resultData, savePath);

                Console.Write($"\rОбработано {i + 1}/{files.Length}...");
            }

            totalStopwatch.Stop();
            Console.WriteLine("\n\nРезультаты:");
            Console.WriteLine($"  Общее время (включая I/O): {totalStopwatch.ElapsedMilliseconds} мс");
            Console.WriteLine($"  Полный GPU-пайплайн (копирование туда-обратно + ядро): {fullGpuPipelineTimeMs} мс (~{fullGpuPipelineTimeMs / (double)files.Length:F2} мс/фото)");
            Console.WriteLine($"  GPU-вычисления (копирование НА GPU + ядро, без обратного копирования): {computeOnlyTimeMs} мс (~{computeOnlyTimeMs / (double)files.Length:F2} мс/фото)");
        }

        public void Dispose()
        {
            accelerator?.Dispose();
            context?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}