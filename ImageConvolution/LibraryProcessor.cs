using System;
using System.IO;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ImageConvolution
{
    public class LibraryProcessor
    {
        public static void ProcessImagesWithImageSharp(string inputDirectory, string outputDirectory)
        {
            if (!Directory.Exists(inputDirectory))
            {
                Console.WriteLine("Ошибка: папка не найдена.");
                return;
            }

            if (!Directory.Exists(outputDirectory))
            {
                Directory.CreateDirectory(outputDirectory);
            }

            ImageFiles.ValidateDirectories(inputDirectory, outputDirectory);
            string[] files = ImageFiles.GetFiles(inputDirectory);

            var options = new ParallelOptions
            {
                MaxDegreeOfParallelism = Math.Min(Environment.ProcessorCount, ImageFiles.GetInFlightLimit(files, 0, 24))
            };
            Parallel.ForEach(files, options, file =>
            {
                string savePath = Path.Combine(outputDirectory, Path.GetFileName(file));
                ImageIO.SaveImageFloat(ConvolveBox(ImageIO.LoadAsGrayscaleFloat(file)), savePath);
            });
        }

        public static float[,] ConvolveBox(float[,] data)
        {
            ConvolutionValidation.Validate(data, Kernels.BlurBoxFloat, EdgeStrategy.Extend);
            using var image = new Image<RgbaVector>(data.GetLength(1), data.GetLength(0));
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < image.Height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < image.Width; x++)
                    {
                        float gray = data[y, x] / 255f;
                        row[x] = new RgbaVector(gray, gray, gray, 1f);
                    }
                }
            });
            image.Mutate(x => x.BoxBlur(1));
            var result = new float[image.Height, image.Width];
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < image.Height; y++)
                {
                    var row = accessor.GetRowSpan(y);
                    for (int x = 0; x < image.Width; x++) result[y, x] = row[x].R * 255f;
                }
            });
            return result;
        }
    }
}
