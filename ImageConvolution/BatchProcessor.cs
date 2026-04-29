using System;
using System.Diagnostics;
using System.IO;
using System.Net;
using System.Runtime.Intrinsics.Arm;
using System.Threading.Tasks;

namespace ImageConvolution
{
    public class BatchProcessor
    {

        public static void ProcessImagesNaiveParallel(string inputDirectory, string outputDirectory, bool useParallelConvolutionInside)
        {
            if (!Directory.Exists(inputDirectory))
            {
                Console.WriteLine("Ошибка: Указанная папка не существует.");
                return;
            }

            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            string[] files = Directory.GetFiles(inputDirectory, "*.jpg");

            Console.WriteLine($"Найдено файлов: {files.Length}");


            Parallel.ForEach(files, currentFile =>
            {
                string fileName = Path.GetFileName(currentFile);

                string savePath = Path.Combine(outputDirectory, fileName);

                double[,] imageData = ImageIO.LoadAsGrayscale(currentFile);

                double[,] result;
                if (useParallelConvolutionInside)
                {
                    result = ParallelConvolutionProcessor.ConvolveParallel(imageData, Kernels.BlurBox);
                }
                else
                {
                    result = ConvolutionProcessor.Convolve(imageData, Kernels.BlurBox);
                }

                ImageIO.SaveImage(result, savePath);
            });
        }
    }
}