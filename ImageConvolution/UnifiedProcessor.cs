using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;

namespace ImageConvolution
{
    public enum ProcessorType { CPU, GPU }

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
        public int CpuWorkers { get; set; } = Math.Max(1, Environment.ProcessorCount / 2);
        public int GpuWorkers { get; set; } = 1;
        public int ReaderThreads { get; set; } = 1; 
        public int WriterThreads { get; set; } = 1; 
        public int QueueCapacity { get; set; } = 4;
        public long MemoryBudgetBytes { get; set; } = 0;
        public ParallelStrategy CpuStrategy { get; set; } = ParallelStrategy.Sequential;
        public float[,] Kernel { get; set; } = Kernels.BlurBoxFloat;
        public EdgeStrategy Strategy { get; set; } = EdgeStrategy.Extend;
    }

    public class UnifiedProcessor : IDisposable
    {
        private readonly UnifiedProcessorConfig config;
        private readonly object sync = new();
        private bool disposed;

        public UnifiedProcessor(UnifiedProcessorConfig config)
        {
            ArgumentNullException.ThrowIfNull(config);
            if (config.CpuWorkers < 0 || config.GpuWorkers < 0 || (config.CpuWorkers == 0 && config.GpuWorkers == 0))
                throw new ArgumentException("Нужен хотя бы один вычислитель", nameof(config));
            if (config.ReaderThreads <= 0 || config.WriterThreads <= 0 || config.QueueCapacity <= 0 || config.MemoryBudgetBytes < 0)
                throw new ArgumentException("Нужны положительные количества читателей, писателей и мест в очереди; бюджет памяти: 0 (авто) или положительное число", nameof(config));
            ConvolutionValidation.ValidateKernel(config.Kernel, config.Strategy);
            if (!Enum.IsDefined(config.CpuStrategy)) throw new ArgumentOutOfRangeException(nameof(config));
            this.config = new UnifiedProcessorConfig
            {
                CpuWorkers = config.CpuWorkers, GpuWorkers = config.GpuWorkers,
                ReaderThreads = config.ReaderThreads, WriterThreads = config.WriterThreads,
                QueueCapacity = config.QueueCapacity, MemoryBudgetBytes = config.MemoryBudgetBytes,
                Kernel = (float[,])config.Kernel.Clone(), Strategy = config.Strategy, CpuStrategy = config.CpuStrategy
            };
        }

        public void ProcessDirectory(string inputDirectory, string outputDirectory)
        {
            lock (sync)
            {
                ObjectDisposedException.ThrowIf(disposed, this);
                ProcessDirectoryAsync(inputDirectory, outputDirectory).GetAwaiter().GetResult();
            }
        }

        private async Task ProcessDirectoryAsync(string inputDirectory, string outputDirectory)
        {
            if (!Directory.Exists(inputDirectory)) return;
            ImageFiles.ValidateDirectories(inputDirectory, outputDirectory);
            Directory.CreateDirectory(outputDirectory);
            var timer = Stopwatch.StartNew();
            var files = ImageFiles.GetFiles(inputDirectory);
            if (files.Length == 0) return;
            int deviceCount = config.GpuWorkers > 0 ? GpuConvolutionProcessor.GetDeviceCount() : 0;
            if (config.GpuWorkers > 0 && deviceCount == 0)
                throw new InvalidOperationException("Запрошены GPU агенты, но CUDA устройства не найдены");
            using var slots = new SemaphoreSlim(ImageFiles.GetInFlightLimit(files, config.MemoryBudgetBytes, 8));
            using var cancellation = new CancellationTokenSource();
            var token = cancellation.Token;
            var inputChannel = Channel.CreateBounded<ProcessingTask>(config.QueueCapacity);
            var outputChannel = Channel.CreateBounded<ProcessingResult>(config.QueueCapacity);
            var filesQueue = new ConcurrentQueue<string>(files);
            Exception? failure = null;
            int processedCount = 0, cpuCount = 0, gpuCount = 0;
            Task Start(Func<Task> action) => Task.Run(async () =>
            {
                try { await action(); }
                catch (OperationCanceledException) when (token.IsCancellationRequested) { }
                catch (Exception ex)
                {
                    Interlocked.CompareExchange(ref failure, ex, null);
                    cancellation.Cancel();
                }
            });
            async Task Complete(Task[] tasks, Action complete)
            {
                try { await Task.WhenAll(tasks); }
                finally { complete(); }
            }
            var readers = Enumerable.Range(0, config.ReaderThreads).Select(_ => Start(async () =>
            {
                while (filesQueue.TryDequeue(out var file))
                {
                    await slots.WaitAsync(token);
                    var data = ImageIO.LoadAsGrayscaleFloat(file);
                    await inputChannel.Writer.WriteAsync(new ProcessingTask
                    {
                        InputPath = file, OutputPath = Path.Combine(outputDirectory, Path.GetFileName(file)), ImageData = data
                    }, token);
                    data = null!;
                }
            })).ToArray();
            var readerCompletion = Complete(readers, () => inputChannel.Writer.TryComplete());
            var workers = new List<Task>();
            for (int i = 0; i < config.GpuWorkers; i++)
            {
                int deviceIndex = i % deviceCount;
                workers.Add(Start(async () =>
                {
                    using var gpu = new GpuConvolutionProcessor(deviceIndex);
                    Console.WriteLine($"GPU агент: устройство {deviceIndex}, {gpu.accelerator.Name}");
                    await foreach (var item in inputChannel.Reader.ReadAllAsync(token))
                    {
                        var result = gpu.ConvolveGpu(item.ImageData, config.Kernel, config.Strategy);
                        item.ImageData = new float[0, 0];
                        await outputChannel.Writer.WriteAsync(new ProcessingResult
                        {
                            OutputPath = item.OutputPath, ResultData = result, ProcessedBy = ProcessorType.GPU
                        }, token);
                        result = null!;
                    }
                }));
            }
            for (int i = 0; i < config.CpuWorkers; i++)
                workers.Add(Start(async () =>
                {
                    await foreach (var item in inputChannel.Reader.ReadAllAsync(token))
                    {
                        var result = ConvolveCpu(item.ImageData, config.Kernel, config.Strategy, config.CpuStrategy);
                        item.ImageData = new float[0, 0];
                        await outputChannel.Writer.WriteAsync(new ProcessingResult
                        {
                            OutputPath = item.OutputPath, ResultData = result, ProcessedBy = ProcessorType.CPU
                        }, token);
                        result = null!;
                    }
                }));
            var workerCompletion = Complete(workers.ToArray(), () => outputChannel.Writer.TryComplete());
            var writers = Enumerable.Range(0, config.WriterThreads).Select(_ => Start(async () =>
            {
                await foreach (var result in outputChannel.Reader.ReadAllAsync(token))
                {
                    ImageIO.SaveImageFloat(result.ResultData, result.OutputPath);
                    result.ResultData = new float[0, 0];
                    Interlocked.Increment(ref processedCount);
                    if (result.ProcessedBy == ProcessorType.CPU) Interlocked.Increment(ref cpuCount);
                    else Interlocked.Increment(ref gpuCount);
                    slots.Release();
                }
            })).ToArray();
            await Task.WhenAll(writers.Append(readerCompletion).Append(workerCompletion));
            if (failure != null) System.Runtime.ExceptionServices.ExceptionDispatchInfo.Capture(failure).Throw();
            Console.WriteLine($"Обработано {processedCount}/{files.Length}; CPU: {cpuCount}, GPU: {gpuCount}; {timer.Elapsed.TotalMilliseconds:F1} мс");
        }

        public static float[,] ConvolveCpu(float[,] image, float[,] kernel, EdgeStrategy strategy, ParallelStrategy parallelStrategy = ParallelStrategy.Sequential)
        {
            return ConvolutionCore.Convolve(image, kernel, strategy, parallelStrategy);
        }

        public void Dispose()
        {
            lock (sync)
            {
                disposed = true;
                GC.SuppressFinalize(this);
            }
        }
    }
}
