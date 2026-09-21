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
        
        private readonly Action<Index2D, ArrayView2D<float, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>, int, int, int> convolutionKernelFloat;
        private MemoryBuffer2D<float, Stride2D.DenseY>? sourceBufferFloat;
        private MemoryBuffer2D<float, Stride2D.DenseY>? targetBufferFloat;

        private readonly Action<Index2D, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>, int, int, int> convolutionKernelByte;
        private MemoryBuffer2D<byte, Stride2D.DenseY>? sourceBufferByte;
        private MemoryBuffer2D<byte, Stride2D.DenseY>? targetBufferByte;

        private MemoryBuffer2D<float, Stride2D.DenseY>? kernelBuffer;
        private int currentWidth = 0;
        private int currentHeight = 0;
        private float[,]? currentKernelData;
        private readonly object sync = new();
        private bool disposed;

        public static int GetDeviceCount()
        {
            using var context = Context.Create(builder => builder.Cuda());
            return context.Devices.Count(d => d.AcceleratorType == AcceleratorType.Cuda);
        }

        public GpuConvolutionProcessor(int deviceIndex = 0)
        {
            context = Context.Create(builder => builder.Cuda().Optimize(OptimizationLevel.O2));
            try
            {
                var devices = context.Devices.Where(d => d.AcceleratorType == AcceleratorType.Cuda).ToArray();
                if (deviceIndex < 0 || deviceIndex >= devices.Length)
                    throw new ArgumentOutOfRangeException(nameof(deviceIndex), $"CUDA устройств: {devices.Length}");
                accelerator = devices[deviceIndex].CreateAccelerator(context);
                convolutionKernelFloat = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<float, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>, int, int, int>(ConvolutionKernelFloat);
                convolutionKernelByte = accelerator.LoadAutoGroupedStreamKernel<Index2D, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<byte, Stride2D.DenseY>, ArrayView2D<float, Stride2D.DenseY>, int, int, int>(ConvolutionKernelByte);
            }
            catch
            {
                accelerator?.Dispose();
                context.Dispose();
                throw;
            }
        }

        static void ConvolutionKernelFloat(
            Index2D index, ArrayView2D<float, Stride2D.DenseY> input, ArrayView2D<float, Stride2D.DenseY> output, ArrayView2D<float, Stride2D.DenseY> kernel, int strategy, int offsetX, int offsetY)
        {
            int x = index.X; int y = index.Y;
            int imgHeight = input.IntExtent.X; int imgWidth = input.IntExtent.Y;
            int kHeight = kernel.IntExtent.X; int kWidth = kernel.IntExtent.Y;

            float sum = 0.0f;
            for (int ky = 0; ky < kHeight; ky++)
            {
                for (int kx = 0; kx < kWidth; kx++)
                {
                    int pixelY = y + offsetY - ky;
                    int pixelX = x + offsetX - kx;
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
                            val = input[pixelY, pixelX];
                    }
                    sum += val * kernel[ky, kx];
                }
            }
            output[y, x] = sum;
        }

        static void ConvolutionKernelByte(
            Index2D index, ArrayView2D<byte, Stride2D.DenseY> input, ArrayView2D<byte, Stride2D.DenseY> output, ArrayView2D<float, Stride2D.DenseY> kernel, int strategy, int offsetX, int offsetY)
        {
            int x = index.X; int y = index.Y;
            int imgHeight = input.IntExtent.X; int imgWidth = input.IntExtent.Y;
            int kHeight = kernel.IntExtent.X; int kWidth = kernel.IntExtent.Y;

            float sum = 0.0f;
            for (int ky = 0; ky < kHeight; ky++)
            {
                for (int kx = 0; kx < kWidth; kx++)
                {
                    int pixelY = y + offsetY - ky;
                    int pixelX = x + offsetX - kx;
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
                            val = input[pixelY, pixelX];
                    }
                    sum += val * kernel[ky, kx];
                }
            }
            sum = sum < 0.0f ? 0.0f : (sum > 255.0f ? 255.0f : sum);
            output[y, x] = (byte)(sum + 0.5f);
        }

        private void EnsureBuffers(int width, int height, float[,] kernelData, bool useByte)
        {
            if (currentWidth != width || currentHeight != height)
            {
                sourceBufferFloat?.Dispose(); sourceBufferFloat = null;
                targetBufferFloat?.Dispose(); targetBufferFloat = null;
                sourceBufferByte?.Dispose(); sourceBufferByte = null;
                targetBufferByte?.Dispose(); targetBufferByte = null;
                currentWidth = width;
                currentHeight = height;
            }
            var extent = new LongIndex2D(height, width);
            if (useByte)
            {
                sourceBufferByte ??= accelerator.Allocate2DDenseY<byte>(extent);
                targetBufferByte ??= accelerator.Allocate2DDenseY<byte>(extent);
            }
            else
            {
                sourceBufferFloat ??= accelerator.Allocate2DDenseY<float>(extent);
                targetBufferFloat ??= accelerator.Allocate2DDenseY<float>(extent);
            }
            if (kernelBuffer == null || kernelBuffer.IntExtent.X != kernelData.GetLength(0) || kernelBuffer.IntExtent.Y != kernelData.GetLength(1))
            {
                kernelBuffer?.Dispose();
                kernelBuffer = null;
                currentKernelData = null;
                kernelBuffer = accelerator.Allocate2DDenseY<float>(new LongIndex2D(kernelData.GetLength(0), kernelData.GetLength(1)));
            }
            if (!KernelEquals(kernelData))
            {
                kernelBuffer.CopyFromCPU(kernelData);
                currentKernelData = (float[,])kernelData.Clone();
            }
        }

        private bool KernelEquals(float[,] kernel)
        {
            if (currentKernelData == null || currentKernelData.GetLength(0) != kernel.GetLength(0) || currentKernelData.GetLength(1) != kernel.GetLength(1)) return false;
            for (int y = 0; y < kernel.GetLength(0); y++)
                for (int x = 0; x < kernel.GetLength(1); x++)
                    if (currentKernelData[y, x] != kernel[y, x]) return false;
            return true;
        }

        public float[,] ConvolveGpu(float[,] image, float[,] kernelData, EdgeStrategy strategy)
        {
            lock (sync)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                ConvolutionValidation.Validate(image, kernelData, strategy);
                int height = image.GetLength(0), width = image.GetLength(1);
                EnsureBuffers(width, height, kernelData, false);
                sourceBufferFloat!.CopyFromCPU(image);
                convolutionKernelFloat(new Index2D(width, height), sourceBufferFloat!.View, targetBufferFloat!.View, kernelBuffer!.View, (int)strategy, kernelData.GetLength(1) / 2, kernelData.GetLength(0) / 2);
                return targetBufferFloat.GetAsArray2D();
            }
        }

        public void ConvolveGpuMathOnly_ForBenchmark(float[,] kernelData, EdgeStrategy strategy)
        {
            lock (sync)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                ConvolutionValidation.ValidateKernel(kernelData, strategy);
                if (sourceBufferFloat == null || targetBufferFloat == null || kernelBuffer == null || !KernelEquals(kernelData))
                    throw new InvalidOperationException("Сначала вызовите ConvolveGpu с этим ядром");
                convolutionKernelFloat(new Index2D(currentWidth, currentHeight), sourceBufferFloat.View, targetBufferFloat.View, kernelBuffer.View, (int)strategy, kernelData.GetLength(1) / 2, kernelData.GetLength(0) / 2);
                accelerator.Synchronize();
            }
        }

        public byte[,] ConvolveGpuWithTelemetry(byte[,] image, float[,] kernelData, EdgeStrategy strategy,
            out double copyToGpuMs, out double kernelExecMs, out double copyFromGpuMs)
        {
            lock (sync)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                ConvolutionValidation.Validate(image, kernelData, strategy);
                int height = image.GetLength(0), width = image.GetLength(1);
                EnsureBuffers(width, height, kernelData, true);
                accelerator.Synchronize();
                var sw = Stopwatch.StartNew();
                sourceBufferByte!.CopyFromCPU(image);
                accelerator.Synchronize();
                copyToGpuMs = sw.Elapsed.TotalMilliseconds;
                sw.Restart();
                convolutionKernelByte(new Index2D(width, height), sourceBufferByte!.View, targetBufferByte!.View, kernelBuffer!.View, (int)strategy, kernelData.GetLength(1) / 2, kernelData.GetLength(0) / 2);
                accelerator.Synchronize();
                kernelExecMs = sw.Elapsed.TotalMilliseconds;
                sw.Restart();
                var result = targetBufferByte.GetAsArray2D();
                copyFromGpuMs = sw.Elapsed.TotalMilliseconds;
                return result;
            }
        }

        public static void ProcessDirectory(string inputDirectory, string outputDirectory, float[,] kernel, EdgeStrategy strategy, bool printStats = true)
        {
            if (!Directory.Exists(inputDirectory)) return;
            ImageFiles.ValidateDirectories(inputDirectory, outputDirectory);
            Directory.CreateDirectory(outputDirectory);
            string[] files = ImageFiles.GetFiles(inputDirectory);
            if (files.Length == 0) return;

            Console.WriteLine($"Найдено файлов для обработки на GPU: {files.Length}");
            using var gpuProcessor = new GpuConvolutionProcessor();
            
            double totalLoadMs = 0, totalCopyToGpuMs = 0, totalKernelMs = 0, totalCopyFromGpuMs = 0, totalSaveMs = 0;
            var totalStopwatch = Stopwatch.StartNew();
            var swStep = new Stopwatch();

            for (int i = 0; i < files.Length; i++)
            {
                swStep.Restart();
                byte[,] imageData = ImageIO.LoadAsGrayscaleByte(files[i]);
                totalLoadMs += swStep.Elapsed.TotalMilliseconds;

                var resultData = gpuProcessor.ConvolveGpuWithTelemetry(
                    imageData, kernel, strategy, 
                    out double copyToGpu, out double kernelExec, out double copyFromGpu);
                
                totalCopyToGpuMs += copyToGpu; totalKernelMs += kernelExec; totalCopyFromGpuMs += copyFromGpu;

                swStep.Restart();
                string fileName = Path.GetFileName(files[i]) ?? $"img_{i}.jpg"; 
                string savePath = Path.Combine(outputDirectory, fileName);
                
                ImageIO.SaveImageByte(resultData, savePath);
                totalSaveMs += swStep.Elapsed.TotalMilliseconds;

                if (printStats) Console.Write($"\rОбработано {i + 1}/{files.Length}...");
            }
            totalStopwatch.Stop();
            
            if (printStats)
            {
                Console.WriteLine("\n\n=======================================================");
                Console.WriteLine("       ДЕТАЛЬНЫЙ СУММАРНЫЙ ПРОФИЛЬ ВРЕМЕНИ   ");
                Console.WriteLine("=======================================================");
                Console.WriteLine($"1. Загрузка диска + Декод JPEG (CPU):   {totalLoadMs:F1} мс");
                Console.WriteLine($"2. Передача ОЗУ -> Видеопамять (PCIe):  {totalCopyToGpuMs:F1} мс");
                Console.WriteLine($"3. Запуск + ядро GPU + синхронизация:   {totalKernelMs:F1} мс");
                Console.WriteLine($"4. Передача Видеопамять -> ОЗУ (PCIe):  {totalCopyFromGpuMs:F1} мс");
                Console.WriteLine($"5. Энкод JPEG + Запись диска (CPU):     {totalSaveMs:F1} мс");
                Console.WriteLine("-------------------------------------------------------");
                Console.WriteLine($"Общее суммарное время по шагам:        {(totalLoadMs + totalCopyToGpuMs + totalKernelMs + totalCopyFromGpuMs + totalSaveMs):F1} мс");
                Console.WriteLine($"Реальное общее время выполнения:       {totalStopwatch.ElapsedMilliseconds} мс");
                Console.WriteLine("=======================================================");
            }
        }

        public void Dispose()
        {
            lock (sync)
            {
                if (disposed) return;
                disposed = true;
                sourceBufferFloat?.Dispose(); targetBufferFloat?.Dispose();
                sourceBufferByte?.Dispose(); targetBufferByte?.Dispose();
                kernelBuffer?.Dispose();
                accelerator.Dispose();
                context.Dispose();
                GC.SuppressFinalize(this);
            }
        }
    }
}
