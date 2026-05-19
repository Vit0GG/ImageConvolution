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

            string[] files = Directory.GetFiles(inputDirectory, "*.jpg");

            Parallel.ForEach(files, file =>
            {
                using Image<Rgba32> image = Image.Load<Rgba32>(file);

                image.Mutate(x => x.GaussianBlur(1f));

                string savePath = Path.Combine(outputDirectory, Path.GetFileName(file));
                image.Save(savePath);
            });
        }
    }
}