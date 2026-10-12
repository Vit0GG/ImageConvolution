using System;
using System.IO;

using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ImageConvolution
{
    public class LibraryProcessor
    {
        // The decoder produces the L8 image consumed by BoxBlur directly, so the
        // folder path does not copy every pixel through intermediate byte[,] arrays.
        public static void ProcessImagesWithImageSharpDirect(string inputDirectory, string outputDirectory)
        {
            if (!Directory.Exists(inputDirectory)) return;
            ImageFiles.ValidateDirectories(inputDirectory, outputDirectory);
            Directory.CreateDirectory(outputDirectory);
            string[] files = ImageFiles.GetFiles(inputDirectory);
            int limit = PerformanceProfiler.Measure("setup.memory_limit", () => ImageFiles.GetInFlightLimit(files, 0, ImageFiles.EstimatedInFlightBytesPerPixel));
            var options = new ParallelOptions { MaxDegreeOfParallelism = Math.Min(Environment.ProcessorCount, limit) };
            PerformanceProfiler.Set("batch", new
            {
                inputDirectory,
                files = files.Length,
                parallelFiles = options.MaxDegreeOfParallelism,
                pixelType = "L8 byte",
                filter = "BoxBlur radius=1",
                grayscale = "ImageSharp L8 decode",
                output = "L8"
            });
            Parallel.ForEach(files, options, file =>
            {
                using var total = PerformanceProfiler.Measure("imagesharp.direct.file_total");
                using var image = PerformanceProfiler.Measure("imagesharp.direct.decode_read", () => Image.Load<L8>(file));
                using (PerformanceProfiler.Measure("imagesharp.direct.blur")) image.Mutate(x => x.BoxBlur(1));
                using (PerformanceProfiler.Measure("imagesharp.direct.encode_write"))
                    image.Save(Path.Combine(outputDirectory, Path.GetFileName(file)));
            });
        }

        public static void ProcessImagesWithImageSharpNative(string inputDirectory, string outputDirectory)
        {
            if (!Directory.Exists(inputDirectory)) return;
            ImageFiles.ValidateDirectories(inputDirectory, outputDirectory);
            Directory.CreateDirectory(outputDirectory);
            string[] files = ImageFiles.GetFiles(inputDirectory);
            int limit = PerformanceProfiler.Measure("setup.memory_limit", () => ImageFiles.GetInFlightLimit(files, 0, ImageFiles.EstimatedInFlightBytesPerPixel));
            var options = new ParallelOptions { MaxDegreeOfParallelism = Math.Min(Environment.ProcessorCount, limit) };
            PerformanceProfiler.Set("batch", new
            {
                inputDirectory,
                files = files.Length,
                parallelFiles = options.MaxDegreeOfParallelism,
                pixelType = "L8 byte",
                filter = "BoxBlur radius=1",
                grayscale = "0.299 R + 0.587 G + 0.114 B; alpha ignored",
                output = "L8"
            });
            Parallel.ForEach(files, options, file =>
            {
                using var total = PerformanceProfiler.Measure("imagesharp.native.file_total");
                using var source = PerformanceProfiler.Measure("imagesharp.native.decode_read", () => Image.Load<Rgba32>(file));
                using var image = PerformanceProfiler.Measure("imagesharp.native.output_allocate", () => new Image<L8>(source.Width, source.Height));
                using (PerformanceProfiler.Measure("imagesharp.native.grayscale"))
                    source.ProcessPixelRows(image, (input, output) =>
                    {
                        for (int y = 0; y < source.Height; y++)
                        {
                            var from = input.GetRowSpan(y);
                            var to = output.GetRowSpan(y);
                            for (int x = 0; x < from.Length; x++)
                                to[x] = new L8(ImageIO.ToByte(0.299 * from[x].R + 0.587 * from[x].G + 0.114 * from[x].B));
                        }
                    });
                using (PerformanceProfiler.Measure("imagesharp.native.blur")) image.Mutate(x => x.BoxBlur(1));
                using (PerformanceProfiler.Measure("imagesharp.native.encode_write"))
                    image.Save(Path.Combine(outputDirectory, Path.GetFileName(file)));
            });
        }

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
                MaxDegreeOfParallelism = Math.Min(Environment.ProcessorCount, ImageFiles.GetInFlightLimit(files, 0, ImageFiles.EstimatedInFlightBytesPerPixel))
            };
            PerformanceProfiler.Set("batch", new
            {
                inputDirectory,
                files = files.Length,
                parallelFiles = options.MaxDegreeOfParallelism,
                pixelType = "byte -> L8 -> byte",
                filter = "BoxBlur radius=1"
            });
            Parallel.ForEach(files, options, file =>
            {
                string savePath = Path.Combine(outputDirectory, Path.GetFileName(file));
                var source = PerformanceProfiler.Measure("reader.load", () => ImageIO.LoadAsGrayscaleByte(file));
                var result = ConvolveBox(source);
                using (PerformanceProfiler.Measure("writer.save")) ImageIO.SaveImageByte(result, savePath);
            });
        }

        public static byte[,] ConvolveBox(byte[,] data)
        {
            ConvolutionValidation.Validate(data, Kernels.BlurBoxFloat, EdgeStrategy.Extend);
            using var total = PerformanceProfiler.Measure("imagesharp.total");
            PerformanceProfiler.DescribeArray(data);
            using var image = PerformanceProfiler.Measure("imagesharp.allocate", () => new Image<L8>(data.GetLength(1), data.GetLength(0)));
            using (PerformanceProfiler.Measure("imagesharp.pack"))
                image.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < image.Height; y++)
                    {
                        var row = accessor.GetRowSpan(y);
                        for (int x = 0; x < image.Width; x++)
                        {
                            row[x] = new L8(data[y, x]);
                        }
                    }
                });
            using (PerformanceProfiler.Measure("imagesharp.blur")) image.Mutate(x => x.BoxBlur(1));
            return PerformanceProfiler.Measure("imagesharp.unpack", () =>
            {
                var result = new byte[image.Height, image.Width];
                image.ProcessPixelRows(accessor =>
                {
                    for (int y = 0; y < image.Height; y++)
                    {
                        var row = accessor.GetRowSpan(y);
                        for (int x = 0; x < image.Width; x++) result[y, x] = row[x].PackedValue;
                    }
                });
                return result;
            });
        }
    }
}