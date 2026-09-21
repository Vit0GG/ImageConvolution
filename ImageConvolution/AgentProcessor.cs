using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.ExceptionServices;

namespace ImageConvolution;

public class ImageToProcess
{
    public string FilePath { get; set; } = "";
    public double[,] ImageData { get; set; } = new double[0, 0];
}

public class ImageResult
{
    public string OriginalFileName { get; set; } = "";
    public double[,] ProcessedData { get; set; } = new double[0, 0];
}

public class AgentProcessor
{
    public static void ProcessImagesWithAgents(string inputDir, string outputDir, int workerCount,
        int queueCapacity = 4, long memoryBudgetBytes = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(workerCount);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(queueCapacity);
        ArgumentOutOfRangeException.ThrowIfNegative(memoryBudgetBytes);
        if (!Directory.Exists(inputDir)) return;
        ImageFiles.ValidateDirectories(inputDir, outputDir);
        Directory.CreateDirectory(outputDir);
        var stopwatch = Stopwatch.StartNew();
        var files = ImageFiles.GetFiles(inputDir);
        using var slots = new SemaphoreSlim(ImageFiles.GetInFlightLimit(files, memoryBudgetBytes, 16));
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
        var reader = Start(() =>
        {
            try
            {
                foreach (var file in files)
                {
                    slots.Wait(token);
                    input.Add(new ImageToProcess { FilePath = file, ImageData = ImageIO.LoadAsGrayscale(file) }, token);
                }
            }
            finally { input.CompleteAdding(); }
        });
        var workers = Enumerable.Range(0, workerCount).Select(_ => Start(() =>
        {
            foreach (var item in input.GetConsumingEnumerable(token))
            {
                var result = new ImageResult
                {
                    OriginalFileName = Path.GetFileName(item.FilePath),
                    ProcessedData = ConvolutionProcessor.Convolve(item.ImageData, Kernels.BlurBox)
                };
                item.ImageData = new double[0, 0];
                output.Add(result, token);
            }
        })).ToArray();
        var completion = Task.WhenAll(workers).ContinueWith(_ => output.CompleteAdding(), TaskScheduler.Default);
        var writer = Start(() =>
        {
            foreach (var result in output.GetConsumingEnumerable(token))
            {
                ImageIO.SaveImage(result.ProcessedData, Path.Combine(outputDir, result.OriginalFileName));
                result.ProcessedData = new double[0, 0];
                slots.Release();
            }
        });
        Task.WaitAll(reader, completion, writer);
        if (failure != null) ExceptionDispatchInfo.Capture(failure).Throw();
        Console.WriteLine($"Обработка конвейером ({workerCount} агентов): {files.Length} файлов за {stopwatch.Elapsed.TotalMilliseconds:F1} мс.");
    }
}
