using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.ExceptionServices;

namespace ImageConvolution;

public class ImageToProcess
{
    public string FilePath { get; set; } = "";
    public byte[,] ImageData { get; set; } = new byte[0, 0];
}

public class ImageResult
{
    public string OriginalFileName { get; set; } = "";
    public byte[,] ProcessedData { get; set; } = new byte[0, 0];
}

public class AgentProcessor
{
    public static void ProcessImagesWithAgents(string inputDir, string outputDir, int workerCount,
        int queueCapacity = 4, long memoryBudgetBytes = 0, int readerCount = 1, int writerCount = 1)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(workerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(queueCapacity);
        ArgumentOutOfRangeException.ThrowIfNegative(memoryBudgetBytes);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(readerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(writerCount);
        if (!Directory.Exists(inputDir)) return;
        ImageFiles.ValidateDirectories(inputDir, outputDir);
        Directory.CreateDirectory(outputDir);
        var stopwatch = Stopwatch.StartNew();
        var files = ImageFiles.GetFiles(inputDir);
        int limit = PerformanceProfiler.Measure("setup.memory_limit", () => ImageFiles.GetInFlightLimit(files, memoryBudgetBytes, ImageFiles.EstimatedInFlightBytesPerPixel));
        PerformanceProfiler.Set("pipeline", new
        {
            inputDir,
            files = files.Length,
            cpuWorkers = workerCount,
            gpuWorkers = 0,
            readers = readerCount,
            writers = writerCount,
            queueCapacity,
            inFlightLimit = limit,
            pixelType = "byte",
            memoryBudgetBytes
        });
        var filesQueue = new ConcurrentQueue<string>(files);
        using var slots = new SemaphoreSlim(limit);
        using var input = new BlockingCollection<ImageToProcess>(queueCapacity);
        using var output = new BlockingCollection<ImageResult>(queueCapacity);
        using var cancellation = new CancellationTokenSource();
        var token = cancellation.Token;
        Exception? failure = null;
        Task Start(Action action) => Task.Factory.StartNew(() =>
        {
            try { action(); }
            catch (OperationCanceledException) when (token.IsCancellationRequested) { }
            catch (Exception ex)
            {
                Interlocked.CompareExchange(ref failure, ex, null);
                cancellation.Cancel();
            }
        }, CancellationToken.None, TaskCreationOptions.LongRunning, TaskScheduler.Default);
        var readers = Enumerable.Range(0, readerCount).Select(_ => Start(() =>
        {
            while (filesQueue.TryDequeue(out var file))
            {
                using (PerformanceProfiler.Measure("reader.wait_slot")) slots.Wait(token);
                var item = new ImageToProcess
                {
                    FilePath = file,
                    ImageData = PerformanceProfiler.Measure("reader.load", () => ImageIO.LoadAsGrayscaleByte(file))
                };
                using (PerformanceProfiler.Measure("reader.enqueue")) input.Add(item, token);
            }
        })).ToArray();
        var readerCompletion = Task.WhenAll(readers).ContinueWith(_ => input.CompleteAdding(), TaskScheduler.Default);
        var workers = Enumerable.Range(0, workerCount).Select(_ => Start(() =>
        {
            foreach (var item in PerformanceProfiler.Consume(input, "cpu.wait_input", token))
            {
                var result = new ImageResult
                {
                    OriginalFileName = Path.GetFileName(item.FilePath),
                    ProcessedData = ConvolutionProcessor.Convolve(item.ImageData, Kernels.BlurBox)
                };
                item.ImageData = new byte[0, 0];
                using (PerformanceProfiler.Measure("cpu.enqueue_result")) output.Add(result, token);
            }
        })).ToArray();
        var completion = Task.WhenAll(workers).ContinueWith(_ => output.CompleteAdding(), TaskScheduler.Default);
        var writers = Enumerable.Range(0, writerCount).Select(_ => Start(() =>
        {
            foreach (var result in PerformanceProfiler.Consume(output, "writer.wait_input", token))
            {
                using (PerformanceProfiler.Measure("writer.save"))
                    ImageIO.SaveImageByte(result.ProcessedData, Path.Combine(outputDir, result.OriginalFileName));
                result.ProcessedData = new byte[0, 0];
                slots.Release();
            }
        })).ToArray();
        Task.WaitAll(writers.Append(readerCompletion).Append(completion).ToArray());
        if (failure != null) ExceptionDispatchInfo.Capture(failure).Throw();
        Console.WriteLine($"Обработка конвейером ({workerCount} вычислителей, {readerCount} читателей, {writerCount} писателей): {files.Length} файлов за {stopwatch.Elapsed.TotalMilliseconds:F1} мс.");
    }
}