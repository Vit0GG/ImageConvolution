using System;
using System.Data;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;
using System.IO;
using System.Threading.Tasks;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;

namespace ImageConvolution
{
    [ExcludeFromCodeCoverage]
    class Program
    {
        static void Main(string[] args)
        {

            Console.WriteLine("=== Программа свёртки изображений ===");
            Console.WriteLine("1. Обработать один файл");
            Console.WriteLine("2. Обработать набор файлов");
            Console.WriteLine("3. Обработать набор файлов агентами");

            string? choice = Console.ReadLine();

            if (choice == "1")
            {
                Console.Write("Введите путь к изображению: ");
                string? inputPath = Console.ReadLine()?.Trim('\"', ' ', '\'');
                if (!File.Exists(inputPath)) return;
                
                double[,] img = ImageIO.LoadAsGrayscale(inputPath);
                double[,] res = ConvolutionProcessor.Convolve(img, Kernels.BlurBox);

                string directory = Path.GetDirectoryName(inputPath)!;
                string fileNameOnly = Path.GetFileNameWithoutExtension(inputPath);
                string extension = Path.GetExtension(inputPath);

                string baseOutputPath = Path.Combine(directory, $"{fileNameOnly}_filtered");

                int counter = 1;
                string finalOutputPath = $"{baseOutputPath}_{counter}{extension}";

                while (File.Exists(finalOutputPath))
                {
                    counter++;
                    finalOutputPath = $"{baseOutputPath}_{counter}{extension}";
                }

                ImageIO.SaveImage(res, finalOutputPath);
                Console.WriteLine($"\nГотово! Изображение сохранено здесь:\n{finalOutputPath}");
            }

            else if (choice == "2")
            {
                Console.WriteLine("Введите исходный путь к папке или перетащите её");
                string? inputDir = Console.ReadLine()?.Trim('\"', ' ', '\'');

                if (string.IsNullOrEmpty(inputDir)) return;

                string outputDir = Path.Combine(Path.GetDirectoryName(inputDir) ?? "", "Processed_Output");


                Console.WriteLine("\n--- ТЕСТ 1: Только внешний параллелизм (Параллельно файлы) ---");
                BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir + "_Seq", false);

                Console.WriteLine("\n--- ТЕСТ 2: Вложенный параллелизм (Параллельно файлы) ---");
                BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir + "_Par", true);
            }

            else if (choice == "3")
            {
                Console.WriteLine("Введите путь к папке:");
                string? inputDir = Console.ReadLine()?.Trim('\"', ' ', '\'');
                if (string.IsNullOrEmpty(inputDir) || !Directory.Exists(inputDir)) return;

                Console.Write("Введите количество агентов для свёртки(от количества ядер на устройстве): ");
                if (!int.TryParse(Console.ReadLine(), out int workerCount))
                {
                    workerCount = Environment.ProcessorCount;
                }

                string outputDir = Path.Combine(Path.GetDirectoryName(inputDir)!, "Agent_Processed_Output");

                AgentProcessor.ProcessImagesWithAgents(inputDir, outputDir, workerCount);
            }

            else
            {
                Console.WriteLine("Неверный выбор.");
            }

        }
    }

    public class ImageIO
    {
        public static double[,] LoadAsGrayscale(string path)
        {
            using Image<Rgba32> image = Image.Load<Rgba32>(path);
            int width = image.Width;
            int height = image.Height;
            double[,] result = new double[height, width];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    Rgba32 pixel = image[x, y];
                    double gray = 0.299 * pixel.R + 0.587 * pixel.G + 0.114 * pixel.B;
                    result[y, x] = gray;
                }
            }
            return result;
        }
        public static void SaveImage(double[,] data, string path)
        {
            int height = data.GetLength(0);
            int width = data.GetLength(1);

            using Image<Rgba32> image = new Image<Rgba32>(width, height);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    double val = data[y, x];
                    byte gray = (byte)Math.Clamp(val, 0, 255);
                    image[x, y] = new Rgba32(gray, gray, gray);
                }
            }
            image.Save(path);
        }
    }

    public class Kernels
    {
        public static double[,] Sharpen = new double[,]
        {
            {  0, -1,  0},
            { -1,  5, -1},
            {  0, -1,  0}
        };
        public static double[,] Laplacian = new double[,]
        {
            {-1, -1, -1},
            {-1,  8, -1},
            {-1, -1, -1}
        };
        public static double[,] BlurBox = new double[,]
        {
            {1.0/9.0, 1.0/9.0, 1.0/9.0},
            {1.0/9.0, 1.0/9.0, 1.0/9.0},
            {1.0/9.0, 1.0/9.0, 1.0/9.0}
        };
    }

    public class ConvolutionProcessor
    {
        public enum EdgeStrategy
        {
            Extend,
            ZeroPadding
        }

        public static double[,] Convolve(double[,] image, double[,] kernel, EdgeStrategy strategy = EdgeStrategy.Extend)
        {
            int imgheight = image.GetLength(0);
            int imgwidth = image.GetLength(1);
            int kheight = kernel.GetLength(0);
            int kwidth = kernel.GetLength(1);

            int offsetY = kheight / 2;
            int offsetX = kwidth / 2;

            double[,] result = new double[imgheight, imgwidth];

            for (int y = 0; y < imgheight; y++)
            {
                for (int x = 0; x < imgwidth; x++)
                {
                    double sum = 0.0;

                    for (int ky = 0; ky < kheight; ky++)
                    {
                        for (int kx = 0; kx < kwidth; kx++)
                        {
                            int pixelY = y + ky - offsetY;
                            int pixelX = x + kx - offsetX;
                            double pixelValue = 0.0;

                            if (strategy == EdgeStrategy.Extend)
                            {
                                if (pixelY < 0) pixelY = 0;
                                if (pixelY >= imgheight) pixelY = imgheight - 1;
                                if (pixelX < 0) pixelX = 0;
                                if (pixelX >= imgwidth) pixelX = imgwidth - 1;

                                pixelValue = image[pixelY, pixelX];
                            }
                            else if (strategy == EdgeStrategy.ZeroPadding)
                            {
                                if (pixelY >= 0 && pixelY < imgheight && pixelX >= 0 && pixelX < imgwidth)
                                {
                                    pixelValue = image[pixelY, pixelX];
                                }
                            }
                            sum += pixelValue * kernel[ky, kx];
                        }
                    }

                    result[y, x] = sum;
                }
            }
            return result;
        }
    }
    public class ParallelConvolutionProcessor
    {
        public enum EdgeStrategy
        {
            Extend,
            ZeroPadding
        }

        public static double[,] ConvolveParallel(double[,] image, double[,] kernal, EdgeStrategy strategy = EdgeStrategy.Extend)
        {
            int imgwidth = image.GetLength(1);
            int imgheight = image.GetLength(0);
            int kwidth = kernal.GetLength(0);
            int kheight = kernal.GetLength(1);

            int offsetX = kwidth / 2;
            int offsetY = kheight / 2;

            double[,] result2 = new double[imgheight, imgwidth];
            Parallel.For(0, imgheight, y =>
            {
                for (int x = 0; x < imgwidth; x++)
                {
                    double sum = 0.0;

                    for (int ky = 0; ky < kheight; ky++)
                    {
                        for (int kx = 0; kx < kwidth; kx++)
                        {
                            int pixelY = y + ky - offsetY;
                            int pixelX = x + kx - offsetX;
                            double pixelValue = 0.0;

                            if (strategy == EdgeStrategy.Extend)
                            {
                                if (pixelY < 0) pixelY = 0;
                                if (pixelY >= imgheight) pixelY = imgheight - 1;
                                if (pixelX < 0) pixelX = 0;
                                if (pixelX >= imgwidth) pixelX = imgwidth - 1;

                                pixelValue = image[pixelY, pixelX];
                            }
                            else if (strategy == EdgeStrategy.ZeroPadding)
                            {
                                if (pixelY >= 0 && pixelY < imgheight && pixelX >= 0 && pixelX < imgwidth)
                                {
                                    pixelValue = image[pixelY, pixelX];
                                }
                            }
                            sum += pixelValue * kernal[ky, kx];
                        }
                    }
                    result2[y, x] = sum;
                }
            });
            return result2;
        }
    }
}