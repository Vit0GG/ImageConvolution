using System;
using System.Data;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;
using System.Diagnostics.CodeAnalysis;

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
            Console.Write("Введите путь к изображению (перетащите файл в консоль): ");

            string? inputPath = Console.ReadLine()?.Trim('\"', ' ', '\'');

            if (string.IsNullOrEmpty(inputPath) || !File.Exists(inputPath))
            {
                Console.WriteLine("Ошибка: Файл не найден! Проверьте путь и попробуйте снова.");
                return;
            }

            string directory = Path.GetDirectoryName(inputPath) ?? "";
            string fileName = Path.GetFileNameWithoutExtension(inputPath);
            string extension = Path.GetExtension(inputPath);
            string outputPath = Path.Combine(directory, fileName + "_filtered" + extension);

            try
            {
                Console.WriteLine("\n1. Загрузка изображения...");
                double[,] image = ImageIO.LoadAsGrayscale(inputPath);

                Console.WriteLine("2. Применение свёртки (Размытие / Blur)...");
                Stopwatch sw = Stopwatch.StartNew();

                double[,] result = ConvolutionProcessor.Convolve(
                    image,
                    Kernels.BlurBox,
                    ConvolutionProcessor.EdgeStrategy.Extend);

                sw.Stop();
                Console.WriteLine("2.1. Применение параллельной свёртки (Размытие / Blur)...");
                Stopwatch sw1 = Stopwatch.StartNew();

                double[,] result2 = ParallelConvolutionProcessor.ConvolveParallel(
                    image,
                    Kernels.BlurBox,
                    ParallelConvolutionProcessor.EdgeStrategy.Extend);

                sw1.Stop();

                Console.WriteLine($"Свёртка завершена за {sw.ElapsedMilliseconds} мс");

                Console.WriteLine($"Параллельная свёртка завершена за {sw1.ElapsedMilliseconds} мс");

                Console.WriteLine("3. Сохранение результата...");
                ImageIO.SaveImage(result, outputPath);

                Console.WriteLine($"\nГотово! Изображение сохранено здесь:\n{outputPath}");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Произошла ошибка: {ex.Message}");
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