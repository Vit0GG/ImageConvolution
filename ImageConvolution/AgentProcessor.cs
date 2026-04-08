using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

using Microsoft.VisualBasic;

namespace ImageConvolution
{
    public class ImageToProcess
    {
        public string FilePath { get; set; } = "";
    }

    public class ImageResult
    {
        public string OriginalFileName { get; set; } = "";
        public double[,] ProcessedData { get; set; } = new double[0, 0];
    }


    public class AgentProcessor
    {
        public static void ProcessImagesWithAgents(string inputDir, string outputDir, int workerCount)
        {
            Console.WriteLine($"Запуск конвейера с {workerCount} агентами");
            if (!Directory.Exists(inputDir)) return;
            Directory.CreateDirectory(outputDir);

            var filesToProcessQueue = new BlockingCollection<ImageToProcess>(100);
            var processedImagesQueue = new BlockingCollection<ImageResult>(100);

            var stopwatch = Stopwatch.StartNew();
            var readerAgent = Task.Run(() =>
            {

                foreach (var file in Directory.GetFiles(inputDir, "*jpg"))
                {
                    filesToProcessQueue.Add(new ImageToProcess { FilePath = file });
                }

                filesToProcessQueue.CompleteAdding();
            });

            var workerAgents = new List<Task>();
            for (int i = 0; i < workerCount; i++)
            {
                var worker = Task.Run(() =>
                {
                    foreach (var imageToProcess in filesToProcessQueue.GetConsumingEnumerable())
                    {
                        double[,] image = ImageIO.LoadAsGrayscale(imageToProcess.FilePath);
                        double[,] processedImage = ConvolutionProcessor.Convolve(image, Kernels.BlurBox);
                        var ResultForQue = new ImageResult
                        {
                            OriginalFileName = Path.GetFileName(imageToProcess.FilePath),
                            ProcessedData = processedImage
                        };
                        processedImagesQueue.Add(ResultForQue);
                    }
                });
                workerAgents.Add(worker);
            }

            var allWorkersTask = Task.WhenAll(workerAgents).ContinueWith(t =>
            {
                processedImagesQueue.CompleteAdding();
            });


            var writerAgent = Task.Run(() =>
            {

                foreach (var result in processedImagesQueue.GetConsumingEnumerable())
                {
                    string savePath = Path.Combine(outputDir, result.OriginalFileName);
                    ImageIO.SaveImage(result.ProcessedData, savePath);
                    Console.WriteLine($"Сохранен файл: {result.OriginalFileName}");
                }
            });

            Task.WaitAll(readerAgent, allWorkersTask, writerAgent);

            stopwatch.Stop();
            Console.WriteLine($"\nОбработка конвейером завершена за {stopwatch.ElapsedMilliseconds} мс.");
        }
    }
}