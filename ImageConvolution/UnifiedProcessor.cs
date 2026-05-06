using System;
using System.Collections.Concurrent;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace ImageConvolution
{
    public enum ProcessorType
    {
        CPU,
        GPU
    }

    public class ProcessingTask
    {
        public required string InputPath { get; set; }
        public required string OutputPath { get; set; }
        public required float[,] ImageData { get; set; }
    }

    public class ProcessingResult
    {
        public required string OutputPath { get; set; }
        public required float[,] ResultData { get; set; }
        public ProcessorType ProcessedBy { get; set; }
    }

    public class UnifiedProcessorConfig
    {
        public int CpuWorkers { get; set; } = Environment.ProcessorCount / 2;
        public int GpuWorkers { get; set; } = 1;
        public int ReaderThreads { get; set; } = 2;
        public int WriterThreads { get; set; } = 2;
        public float[,] Kernel { get; set; } = Kernels.BlurBoxFloat;
        public EdgeStrategy Strategy { get; set; } = EdgeStrategy.Extend;
    }

    public class UnifiedProcessor : IDisposable
    {
        private readonly UnifiedProcessorConfig config;
        private readonly BlockingCollection<ProcessingTask> readQueue;
        private readonly BlockingCollection<ProcessingResult> writeQueue;
        private readonly GpuConvolutionProcessor[] gpuProcessors;
        private readonly CancellationTokenSource cts;

        public UnifiedProcessor(UnifiedProcessorConfig config)
        {
            this.config = config;
            readQueue = new BlockingCollection<ProcessingTask>(boundedCapacity: 20);
            writeQueue = new BlockingCollection<ProcessingResult>(boundedCapacity: 20);
            cts = new CancellationTokenSource();

            gpuProcessors = new GpuConvolutionProcessor[config.GpuWorkers];
            for (int i = 0; i < config.GpuWorkers; i++)
            {
                gpuProcessors[i] = new GpuConvolutionProcessor();
            }
        }

        public void ProcessDirectory(string inputDirectory, string outputDirectory)
        {
            if (!Directory.Exists(inputDirectory))
            {
                Console.WriteLine("Ошибка: папка не существует");
                return;
            }

            Directory.CreateDirectory(outputDirectory);

            string[] files = Directory.GetFiles(inputDirectory, "*.jpg").OrderBy(f => f).ToArray();
            if (files.Length == 0)
            {
                Console.WriteLine("Файлы .jpg не найдены");
                return;
            }

            Console.WriteLine($"\n=== Unified Processing ===");
            Console.WriteLine($"Файлов: {files.Length}");
            Console.WriteLine($"CPU воркеров: {config.CpuWorkers}");
            Console.WriteLine($"GPU воркеров: {config.GpuWorkers}");
            Console.WriteLine($"Читателей: {config.ReaderThreads}");
            Console.WriteLine($"Писателей: {config.WriterThreads}");

            var totalStopwatch = Stopwatch.StartNew();
            int processedCount = 0;
            int cpuCount = 0;
            int gpuCount = 0;
            object statsLock = new object();

            var readers = new Task[config.ReaderThreads];
            for (int i = 0; i < config.ReaderThreads; i++)
            {
                int threadId = i;
                readers[i] = Task.Run(() => ReaderAgent(files, outputDirectory, threadId));
            }

            var cpuWorkers = new Task[config.CpuWorkers];
            for (int i = 0; i < config.CpuWorkers; i++)
            {
                int workerId = i;
                cpuWorkers[i] = Task.Run(() => CpuWorkerAgent(workerId, ref cpuCount, statsLock));
            }

            var gpuWorkers = new Task[config.GpuWorkers];
            for (int i = 0; i < config.GpuWorkers; i++)
            {
                int workerId = i;
                gpuWorkers[i] = Task.Run(() => GpuWorkerAgent(workerId, ref gpuCount, statsLock));
            }

            var writers = new Task[config.WriterThreads];
            for (int i = 0; i < config.WriterThreads; i++)
            {
                int threadId = i;
                writers[i] = Task.Run(() => WriterAgent(threadId, ref processedCount, files.Length));
            }

            Task.WaitAll(readers);
            readQueue.CompleteAdding();

            Task.WaitAll(cpuWorkers.Concat(gpuWorkers).ToArray());
            writeQueue.CompleteAdding();

            Task.WaitAll(writers);

            totalStopwatch.Stop();

            Console.WriteLine($"\n\n=== Результаты ===");
            Console.WriteLine($"Обработано файлов: {processedCount}");
            Console.WriteLine($"  CPU обработал: {cpuCount}");
            Console.WriteLine($"  GPU обработал: {gpuCount}");
            Console.WriteLine($"Общее время: {totalStopwatch.ElapsedMilliseconds} мс");
            Console.WriteLine($"Среднее время на файл: {totalStopwatch.ElapsedMilliseconds / (double)files.Length:F2} мс");
        }

        private void ReaderAgent(string[] files, string outputDirectory, int threadId)
        {
            int filesPerThread = (int)Math.Ceiling(files.Length / (double)config.ReaderThreads);
            int startIdx = threadId * filesPerThread;
            int endIdx = Math.Min(startIdx + filesPerThread, files.Length);

            for (int i = startIdx; i < endIdx; i++)
            {
                if (cts.Token.IsCancellationRequested) break;

                try
                {
                    string inputPath = files[i];
                    float[,] imageData = ImageIO.LoadAsGrayscaleFloat(inputPath);
                    string fileName = Path.GetFileName(inputPath);
                    string outputPath = Path.Combine(outputDirectory, fileName);

                    var task = new ProcessingTask
                    {
                        InputPath = inputPath,
                        OutputPath = outputPath,
                        ImageData = imageData
                    };

                    readQueue.Add(task, cts.Token);
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\nОшибка чтения {files[i]}: {ex.Message}");
                }
            }
        }

        private void CpuWorkerAgent(int workerId, ref int counter, object lockObj)
        {
            foreach (var task in readQueue.GetConsumingEnumerable(cts.Token))
            {
                try
                {
                    float[,] result = ConvolveCpu(task.ImageData, config.Kernel, config.Strategy);

                    var processingResult = new ProcessingResult
                    {
                        OutputPath = task.OutputPath,
                        ResultData = result,
                        ProcessedBy = ProcessorType.CPU
                    };

                    writeQueue.Add(processingResult, cts.Token);

                    lock (lockObj)
                    {
                        counter++;
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\nОшибка CPU обработки: {ex.Message}");
                }
            }
        }

        private void GpuWorkerAgent(int workerId, ref int counter, object lockObj)
        {
            var gpuProcessor = gpuProcessors[workerId];

            foreach (var task in readQueue.GetConsumingEnumerable(cts.Token))
            {
                try
                {
                    float[,] result = gpuProcessor.ConvolveGpu(task.ImageData, config.Kernel, config.Strategy);

                    var processingResult = new ProcessingResult
                    {
                        OutputPath = task.OutputPath,
                        ResultData = result,
                        ProcessedBy = ProcessorType.GPU
                    };

                    writeQueue.Add(processingResult, cts.Token);

                    lock (lockObj)
                    {
                        counter++;
                    }
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\nОшибка GPU обработки: {ex.Message}");
                }
            }
        }

        private void WriterAgent(int threadId, ref int counter, int total)
        {
            foreach (var result in writeQueue.GetConsumingEnumerable(cts.Token))
            {
                try
                {
                    ImageIO.SaveImageFloat(result.ResultData, result.OutputPath);

                    int current = Interlocked.Increment(ref counter);
                    Console.Write($"\rОбработано: {current}/{total} (последний: {result.ProcessedBy})   ");
                }
                catch (OperationCanceledException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"\nОшибка записи {result.OutputPath}: {ex.Message}");
                }
            }
        }

        private float[,] ConvolveCpu(float[,] image, float[,] kernel, EdgeStrategy strategy)
        {
            int height = image.GetLength(0);
            int width = image.GetLength(1);
            int kHeight = kernel.GetLength(0);
            int kWidth = kernel.GetLength(1);
            int offsetY = kHeight / 2;
            int offsetX = kWidth / 2;

            float[,] result = new float[height, width];

            Parallel.For(0, height, y =>
            {
                for (int x = 0; x < width; x++)
                {
                    float sum = 0.0f;

                    for (int ky = 0; ky < kHeight; ky++)
                    {
                        for (int kx = 0; kx < kWidth; kx++)
                        {
                            int pixelY = y + ky - offsetY;
                            int pixelX = x + kx - offsetX;
                            float val = 0.0f;

                            if (strategy == EdgeStrategy.Extend)
                            {
                                pixelY = Math.Clamp(pixelY, 0, height - 1);
                                pixelX = Math.Clamp(pixelX, 0, width - 1);
                                val = image[pixelY, pixelX];
                            }
                            else if (strategy == EdgeStrategy.ZeroPadding)
                            {
                                if (pixelY >= 0 && pixelY < height && pixelX >= 0 && pixelX < width)
                                {
                                    val = image[pixelY, pixelX];
                                }
                            }

                            sum += val * kernel[ky, kx];
                        }
                    }

                    result[y, x] = sum;
                }
            });

            return result;
        }

        public void Dispose()
        {
            cts.Cancel();
            readQueue?.Dispose();
            writeQueue?.Dispose();

            if (gpuProcessors != null)
            {
                foreach (var gpu in gpuProcessors)
                {
                    gpu?.Dispose();
                }
            }

            cts?.Dispose();
        }
    }
}