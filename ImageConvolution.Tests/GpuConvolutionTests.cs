using System;
using System.IO;

using Xunit;

namespace ImageConvolution.Tests
{
    public class GpuConvolutionTests : IDisposable
    {
        private const string TestDirInput = "gpu_test_input";
        private const string TestDirOutput = "gpu_test_output";

        public GpuConvolutionTests()
        {
            Cleanup();
        }

        private void Cleanup()
        {
            if (Directory.Exists(TestDirInput)) Directory.Delete(TestDirInput, true);
            if (Directory.Exists(TestDirOutput)) Directory.Delete(TestDirOutput, true);
        }

        public void Dispose()
        {
            Cleanup();
            GC.SuppressFinalize(this);
        }

        [Fact]
        public void GpuProcessor_CanBeConstructedAndDisposed()
        {
            using var gpuProcessor = new GpuConvolutionProcessor();
            Assert.NotNull(gpuProcessor);
        }

        [Fact]
        public void ConvolveGpu_ExtendStrategy_ProducesCorrectCenterPixel()
        {
            using var gpuProcessor = new GpuConvolutionProcessor();
            var image = new float[5, 5];
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    image[i, j] = 100f;

            var result = gpuProcessor.ConvolveGpu(image, Kernels.BlurBoxFloat, EdgeStrategy.Extend);

            Assert.Equal(100, Math.Round(result[2, 2]));
        }

        [Fact]
        public void ConvolveGpu_ZeroPaddingStrategy_AffectsCornerPixel()
        {
            using var gpuProcessor = new GpuConvolutionProcessor();
            var image = new float[5, 5];
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    image[i, j] = 100f;

            var result = gpuProcessor.ConvolveGpu(image, Kernels.BlurBoxFloat, EdgeStrategy.ZeroPadding);

            Assert.True(result[0, 0] < 50);
        }

        [Fact]
        public void ProcessDirectory_ProcessesFilesCorrectly()
        {
            CreateTestImageFloat(TestDirInput, "test1.jpg", 10, 10);
            CreateTestImageFloat(TestDirInput, "test2.jpg", 10, 10);

            GpuConvolutionProcessor.ProcessDirectory(TestDirInput, TestDirOutput, Kernels.BlurBoxFloat, EdgeStrategy.Extend);

            Assert.True(Directory.Exists(TestDirOutput));
            Assert.Equal(2, Directory.GetFiles(TestDirOutput).Length);
            Assert.True(File.Exists(Path.Combine(TestDirOutput, "test1.jpg")));
            Assert.True(File.Exists(Path.Combine(TestDirOutput, "test2.jpg")));
        }

        [Fact]
        public void ProcessDirectory_SkipsMismatchedSizeFiles()
        {
            CreateTestImageFloat(TestDirInput, "test_10x10.jpg", 10, 10);
            CreateTestImageFloat(TestDirInput, "test_20x20.jpg", 20, 20);

            GpuConvolutionProcessor.ProcessDirectory(TestDirInput, TestDirOutput, Kernels.BlurBoxFloat, EdgeStrategy.Extend);

            Assert.True(Directory.Exists(TestDirOutput));
            Assert.Single(Directory.GetFiles(TestDirOutput));
            Assert.True(File.Exists(Path.Combine(TestDirOutput, "test_10x10.jpg")));
            Assert.False(File.Exists(Path.Combine(TestDirOutput, "test_20x20.jpg")));
        }

        [Fact]
        public void ProcessDirectory_HandlesNonExistentInputDirectory()
        {
            GpuConvolutionProcessor.ProcessDirectory("non_existent_dir_12345", TestDirOutput, Kernels.BlurBoxFloat, EdgeStrategy.Extend);

            Assert.False(Directory.Exists(TestDirOutput));
        }

        [Fact]
        public void ProcessDirectory_HandlesEmptyInputDirectory()
        {
            Directory.CreateDirectory(TestDirInput);

            GpuConvolutionProcessor.ProcessDirectory(TestDirInput, TestDirOutput, Kernels.BlurBoxFloat, EdgeStrategy.Extend);

            Assert.True(Directory.Exists(TestDirOutput));
            Assert.Empty(Directory.GetFiles(TestDirOutput));
        }

        private void CreateTestImageFloat(string directory, string filename, int width, int height)
        {
            Directory.CreateDirectory(directory);
            var dummyPixelData = new float[height, width];
            for (int y = 0; y < height; y++)
                for (int x = 0; x < width; x++)
                    dummyPixelData[y, x] = 128f;

            string fullPath = Path.Combine(directory, filename);
            ImageIO.SaveImageFloat(dummyPixelData, fullPath);
        }
    }
}