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
}