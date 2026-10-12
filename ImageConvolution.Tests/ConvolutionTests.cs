using System;
using System.IO;

using ImageConvolution;

using Xunit;

namespace ImageConvolution.Tests;

public class ConvolutionTests
{
    [Theory]
    [InlineData(1, 1, 32)]
    [InlineData(2, 3, 2)]
    [InlineData(35, 33, 32)]
    [InlineData(17, 9, 1)]
    public void LocalityTraversalsMatchForAllPixels(int width, int height, int blockSize)
    {
        var image = new byte[height, width];
        var random = new Random(42);
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++) image[y, x] = (byte)random.Next(256);
        float[,] kernel = { { 1, -2, 3 }, { 4, 0.5f, -6 }, { 7, 8, 9 } };
        foreach (var edge in new[] { EdgeStrategy.Extend, EdgeStrategy.ZeroPadding })
        {
            var expected = ConvolutionProcessor.Convolve(image, kernel, edge);
            foreach (var traversal in Enum.GetValues<SequentialTraversal>())
            {
                var actual = LocalityExperiment.Convolve(image, kernel, edge, traversal, blockSize);
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++) Assert.Equal(expected[y, x], actual[y, x]);
            }
            var parallel = ParallelConvolutionProcessor.ConvolveParallelByColumns(image, kernel, edge);
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++) Assert.Equal(expected[y, x], parallel[y, x]);
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(2, 3)]
    [InlineData(17, 9)]
    public void NativeImageSharpMatchesWrapperAndCpu(int width, int height)
    {
        string root = Path.Combine(Path.GetTempPath(), "ImageSharpNative-" + Guid.NewGuid().ToString("N"));
        string input = Path.Combine(root, "input");
        string native = Path.Combine(root, "native");
        string wrapper = Path.Combine(root, "wrapper");
        Directory.CreateDirectory(input);
        try
        {
            string path = Path.Combine(input, "test.png");
            using (var image = new SixLabors.ImageSharp.Image<SixLabors.ImageSharp.PixelFormats.Rgba32>(width, height))
            {
                for (int y = 0; y < height; y++)
                    for (int x = 0; x < width; x++)
                        image[x, y] = new((byte)((x * 73 + y * 19) % 256), (byte)((x * 11 + y * 97) % 256),
                            (byte)((x * 31 + y * 47) % 256), (byte)((x * 17 + y * 23) % 256));
                SixLabors.ImageSharp.ImageExtensions.Save(image, path);
            }
            using var session = new PerformanceProfiler.Session();
            LibraryProcessor.ProcessImagesWithImageSharpNative(input, native);
            var profile = session.Capture();
            Assert.Equal(1L, Assert.Single(profile.Stages, s => s.Name == "imagesharp.native.blur").Count);
            LibraryProcessor.ProcessImagesWithImageSharp(input, wrapper);
            var actual = ImageIO.LoadAsGrayscaleByte(Path.Combine(native, "test.png"));
            var previous = ImageIO.LoadAsGrayscaleByte(Path.Combine(wrapper, "test.png"));
            var expected = UnifiedProcessor.ConvolveCpu(ImageIO.LoadAsGrayscaleByte(path), Kernels.BlurBoxFloat, EdgeStrategy.Extend);
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                {
                    Assert.InRange(Math.Abs(actual[y, x] - previous[y, x]), 0, 1);
                    Assert.InRange(Math.Abs(actual[y, x] - ImageIO.ToByte(expected[y, x])), 0, 1);
                }
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, true);
        }
    }

    [Fact]
    public void Test_ImageIO_SaveAndLoad()
    {
        string testPath = "test_image.png";
        byte[,] originalData = new byte[2, 2] { { 0, 255 }, { 128, 64 } };

        try
        {
            ImageIO.SaveImageByte(originalData, testPath);
            Assert.True(File.Exists(testPath));

            byte[,] loadedData = ImageIO.LoadAsGrayscaleByte(testPath);

            Assert.Equal(2, loadedData.GetLength(0));
            Assert.Equal(2, loadedData.GetLength(1));
        }
        finally
        {
            if (File.Exists(testPath)) File.Delete(testPath);
        }
    }

    [Fact]
    public void Test_Convolve_ExtendStrategy()
    {
        byte[,] image = { { 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 } };
        byte[,] result = ConvolutionProcessor.Convolve(image, Kernels.BlurBox, EdgeStrategy.Extend);
        Assert.Equal(10, result[1, 1]);
    }

    [Fact]
    public void Test_Convolve_ZeroPaddingStrategy()
    {
        byte[,] image = { { 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 } };
        byte[,] result = ConvolutionProcessor.Convolve(image, Kernels.BlurBox, EdgeStrategy.ZeroPadding);
        Assert.True(result[0, 0] < 10);
    }

    [Fact]
    public void Test_ConvolveParallel()
    {
        byte[,] image = { { 100, 100, 100 }, { 100, 100, 100 }, { 100, 100, 100 } };
        byte[,] result = ParallelConvolutionProcessor.ConvolveParallel(image, Kernels.BlurBox, EdgeStrategy.Extend);
        Assert.Equal(100, result[1, 1]);
    }

    [Fact]
    public void Test_KernelsExist()
    {
        Assert.NotNull(Kernels.BlurBox);
        Assert.NotNull(Kernels.Sharpen);
        Assert.NotNull(Kernels.Laplacian);
    }

    [Fact]
    public void Test_ConvolveParallel_ZeroPadding()
    {
        byte[,] image = {
                { 10, 10, 10 },
                { 10, 10, 10 },
                { 10, 10, 10 }
            };

        byte[,] result = ParallelConvolutionProcessor.ConvolveParallel(
            image,
            Kernels.BlurBox,
            EdgeStrategy.ZeroPadding);

        Assert.True(result[0, 0] < 10);
    }

    private void CreateTestImage(string directory, string filename)
    {
        Directory.CreateDirectory(directory);
        byte[,] dummyPixelData = new byte[2, 2] { { 128, 128 }, { 128, 128 } };
        string fullPath = Path.Combine(directory, filename);
        ImageIO.SaveImageByte(dummyPixelData, fullPath);
    }

    [Fact]
    public void Test_BatchProcessor_Naive_Sequential()
    {
        string inputDir = "test_input_naive_seq";
        string outputDir = "test_output_naive_seq";

        try
        {
            CreateTestImage(inputDir, "test1.jpg");

            BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir, ParallelStrategy.Sequential);

            Assert.True(Directory.Exists(outputDir));
            Assert.Single(Directory.GetFiles(outputDir));
        }
        finally
        {
            if (Directory.Exists(inputDir)) Directory.Delete(inputDir, true);
            if (Directory.Exists(outputDir)) Directory.Delete(outputDir, true);
        }
    }

    [Fact]
    public void Test_BatchProcessor_Naive_Parallel()
    {
        string inputDir = "test_input_naive_par";
        string outputDir = "test_output_naive_par";

        try
        {
            CreateTestImage(inputDir, "test1.jpg");

            BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir, ParallelStrategy.ParallelByRows);

            Assert.True(Directory.Exists(outputDir));
            Assert.Single(Directory.GetFiles(outputDir));
        }
        finally
        {
            if (Directory.Exists(inputDir)) Directory.Delete(inputDir, true);
            if (Directory.Exists(outputDir)) Directory.Delete(outputDir, true);
        }
    }

    [Fact]
    public void Test_AgentProcessor_Success()
    {
        string inputDir = "test_input_agents";
        string outputDir = "test_output_agents";

        try
        {
            CreateTestImage(inputDir, "test1.jpg");
            CreateTestImage(inputDir, "test2.jpg");

            AgentProcessor.ProcessImagesWithAgents(inputDir, outputDir, workerCount: 2);

            Assert.True(Directory.Exists(outputDir));
            Assert.Equal(2, Directory.GetFiles(outputDir).Length);
        }
        finally
        {
            if (Directory.Exists(inputDir)) Directory.Delete(inputDir, true);
            if (Directory.Exists(outputDir)) Directory.Delete(outputDir, true);
        }
    }

    [Fact]
    public void Test_BatchProcessor_NonExistentFolder()
    {
        string nonExistentDir = "non_existent_folder_abc_123";
        string outputDir = "output_abc";

        BatchProcessor.ProcessImagesNaiveParallel(nonExistentDir, outputDir, ParallelStrategy.Sequential);

        Assert.False(Directory.Exists(outputDir));
    }

    [Fact]
    public void Test_AgentProcessor_NonExistentFolder()
    {
        string nonExistentDir = "non_existent_folder_xyz_999";
        string outputDir = "output_xyz";

        AgentProcessor.ProcessImagesWithAgents(nonExistentDir, outputDir, 2);

        Assert.False(Directory.Exists(outputDir));
    }

    [Fact]
    public void Test_CustomKernelSize_5x5()
    {
        byte[,] image = new byte[10, 10];
        for (int i = 0; i < 10; i++)
            for (int j = 0; j < 10; j++) image[i, j] = 100;

        float[,] kernel5x5 = new float[5, 5];
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++) kernel5x5[i, j] = 1f / 25f;

        byte[,] result = ConvolutionProcessor.Convolve(image, kernel5x5, EdgeStrategy.Extend);
        Assert.Equal(100, result[5, 5]);
    }

    [Fact]
    public void Test_SharpenAndLaplacianKernels()
    {
        byte[,] image = { { 50, 50, 50 }, { 50, 50, 50 }, { 50, 50, 50 } };

        var resLapl = ConvolutionProcessor.Convolve(image, Kernels.Laplacian);
        var resSharp = ConvolutionProcessor.Convolve(image, Kernels.Sharpen);

        Assert.Equal(0, resLapl[1, 1]);
        Assert.Equal(50, resSharp[1, 1]);
    }

    private void CreateLargeTestImage(string directory, string filename)
    {
        Directory.CreateDirectory(directory);
        byte[,] dummyPixelData = new byte[20, 20];
        for (int y = 0; y < 20; y++)
            for (int x = 0; x < 20; x++)
                dummyPixelData[y, x] = 128;

        string fullPath = Path.Combine(directory, filename);
        ImageIO.SaveImageByte(dummyPixelData, fullPath);
    }

    [Fact]
    public void Test_LibraryProcessor_Parallel_Success()
    {
        string inputDir = "test_input_lib_par";
        string outputDir = "test_output_lib_par";
        try
        {
            CreateLargeTestImage(inputDir, "lib_test.jpg");

            LibraryProcessor.ProcessImagesWithImageSharp(inputDir, outputDir);

            Assert.True(Directory.Exists(outputDir));
            Assert.Single(Directory.GetFiles(outputDir));
        }
        finally
        {
            if (Directory.Exists(inputDir)) Directory.Delete(inputDir, true);
            if (Directory.Exists(outputDir)) Directory.Delete(outputDir, true);
        }
    }
    [Fact]
    public void Test_BatchProcessor_EmptyFolder()
    {
        string root = Path.Combine(Path.GetTempPath(), "imageconv-empty-batch-" + Guid.NewGuid().ToString("N"));
        string input = Path.Combine(root, "input");
        string output = Path.Combine(root, "output");
        try
        {
            Directory.CreateDirectory(input);
            BatchProcessor.ProcessImagesNaiveParallel(input, output, ParallelStrategy.Sequential);
            Assert.Empty(Directory.GetFiles(output));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public void Test_AgentProcessor_EmptyFolder()
    {
        string root = Path.Combine(Path.GetTempPath(), "imageconv-empty-agent-" + Guid.NewGuid().ToString("N"));
        string input = Path.Combine(root, "input");
        string output = Path.Combine(root, "output");
        try
        {
            Directory.CreateDirectory(input);
            AgentProcessor.ProcessImagesWithAgents(input, output, 2);
            Assert.Empty(Directory.GetFiles(output));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public void Test_LibraryProcessor_EmptyFolder()
    {
        string root = Path.Combine(Path.GetTempPath(), "imageconv-empty-library-" + Guid.NewGuid().ToString("N"));
        string input = Path.Combine(root, "input");
        string output = Path.Combine(root, "output");
        try
        {
            Directory.CreateDirectory(input);
            LibraryProcessor.ProcessImagesWithImageSharp(input, output);
            Assert.Empty(Directory.GetFiles(output));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    [Fact]
    public void Test_LibraryProcessor_NonExistentFolder()
    {
        string root = Path.Combine(Path.GetTempPath(), "imageconv-missing-library-" + Guid.NewGuid().ToString("N"));
        LibraryProcessor.ProcessImagesWithImageSharp(Path.Combine(root, "input"), Path.Combine(root, "output"));
        Assert.False(Directory.Exists(root));
    }

    [Fact]
    public void Test_BatchProcessor_CorruptImage()
    {
        string root = Path.Combine(Path.GetTempPath(), "imageconv-corrupt-batch-" + Guid.NewGuid().ToString("N"));
        string input = Path.Combine(root, "input");
        string output = Path.Combine(root, "output");
        try
        {
            Directory.CreateDirectory(input);
            File.WriteAllText(Path.Combine(input, "corrupt.jpg"), "bad data");
            Assert.Throws<SixLabors.ImageSharp.UnknownImageFormatException>(
                () => BatchProcessor.ProcessImagesNaiveParallel(input, output, ParallelStrategy.Sequential));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

}