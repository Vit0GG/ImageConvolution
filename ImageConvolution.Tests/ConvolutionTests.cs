using System;
using System.IO;

using ImageConvolution;

using Xunit;

namespace ImageConvolution.Tests;

public class ConvolutionTests
{
    [Fact]
    public void Test_ImageIO_SaveAndLoad()
    {
        string testPath = "test_image.png";
        double[,] originalData = new double[2, 2] { { 0, 255 }, { 128, 64 } };

        try
        {
            ImageIO.SaveImage(originalData, testPath);
            Assert.True(File.Exists(testPath));

            double[,] loadedData = ImageIO.LoadAsGrayscale(testPath);

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
        double[,] image = { { 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 } };
        double[,] result = ConvolutionProcessor.Convolve(image, Kernels.BlurBox, ConvolutionProcessor.EdgeStrategy.Extend);
        Assert.Equal(10, Math.Round(result[1, 1]));
    }

    [Fact]
    public void Test_Convolve_ZeroPaddingStrategy()
    {
        double[,] image = { { 10, 10, 10 }, { 10, 10, 10 }, { 10, 10, 10 } };
        double[,] result = ConvolutionProcessor.Convolve(image, Kernels.BlurBox, ConvolutionProcessor.EdgeStrategy.ZeroPadding);
        Assert.True(result[0, 0] < 10);
    }

    [Fact]
    public void Test_ConvolveParallel()
    {
        double[,] image = { { 100, 100, 100 }, { 100, 100, 100 }, { 100, 100, 100 } };
        double[,] result = ParallelConvolutionProcessor.ConvolveParallel(image, Kernels.BlurBox, ParallelConvolutionProcessor.EdgeStrategy.Extend);
        Assert.Equal(100, Math.Round(result[1, 1]));
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
        double[,] image = {
                { 10, 10, 10 },
                { 10, 10, 10 },
                { 10, 10, 10 }
            };

        double[,] result = ParallelConvolutionProcessor.ConvolveParallel(
            image,
            Kernels.BlurBox,
            ParallelConvolutionProcessor.EdgeStrategy.ZeroPadding);

        Assert.True(result[0, 0] < 10);
    }

        private void CreateTestImage(string directory, string filename)
    {
        Directory.CreateDirectory(directory);
        double[,] dummyPixelData = new double[2, 2] { { 128, 128 }, { 128, 128 } };
        string fullPath = Path.Combine(directory, filename);
        ImageIO.SaveImage(dummyPixelData, fullPath);
    }

    [Fact]
    public void Test_BatchProcessor_Naive_Sequential()
    {
        string inputDir = "test_input_naive_seq";
        string outputDir = "test_output_naive_seq";

        try
        {
            CreateTestImage(inputDir, "test1.jpg");

            BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir, false);

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

            BatchProcessor.ProcessImagesNaiveParallel(inputDir, outputDir, true);

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

        BatchProcessor.ProcessImagesNaiveParallel(nonExistentDir, outputDir, false);
        
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

}