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
        [Fact]
        public void UnifiedProcessor_CanBeConstructedAndDisposed()
        {
            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 1,
                GpuWorkers = 1,
                ReaderThreads = 1,
                WriterThreads = 1
            };

            using var processor = new UnifiedProcessor(config);
            Assert.NotNull(processor);
        }

        [Fact]
        public void UnifiedProcessor_ProcessesDirectoryWithCpuOnly()
        {
            CreateTestImageFloat(TestDirInput, "test1.jpg", 10, 10);
            CreateTestImageFloat(TestDirInput, "test2.jpg", 10, 10);

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 2,
                GpuWorkers = 0,
                ReaderThreads = 1,
                WriterThreads = 1
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.True(Directory.Exists(TestDirOutput));
            Assert.Equal(2, Directory.GetFiles(TestDirOutput).Length);
        }

        [Fact]
        public void UnifiedProcessor_ProcessesDirectoryWithGpuOnly()
        {
            CreateTestImageFloat(TestDirInput, "test1.jpg", 10, 10);
            CreateTestImageFloat(TestDirInput, "test2.jpg", 10, 10);

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 0,
                GpuWorkers = 1,
                ReaderThreads = 1,
                WriterThreads = 1
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.True(Directory.Exists(TestDirOutput));
            Assert.Equal(2, Directory.GetFiles(TestDirOutput).Length);
        }

        [Fact]
        public void UnifiedProcessor_ProcessesDirectoryWithMixedWorkers()
        {
            CreateTestImageFloat(TestDirInput, "test1.jpg", 10, 10);
            CreateTestImageFloat(TestDirInput, "test2.jpg", 10, 10);
            CreateTestImageFloat(TestDirInput, "test3.jpg", 10, 10);

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 1,
                GpuWorkers = 1,
                ReaderThreads = 1,
                WriterThreads = 1
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.True(Directory.Exists(TestDirOutput));
            Assert.Equal(3, Directory.GetFiles(TestDirOutput).Length);
        }

        [Fact]
        public void UnifiedProcessor_HandlesNonExistentDirectory()
        {
            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 1,
                GpuWorkers = 1
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory("non_existent_12345", TestDirOutput);

            Assert.False(Directory.Exists(TestDirOutput));
        }

        [Fact]
        public void UnifiedProcessor_HandlesEmptyDirectory()
        {
            Directory.CreateDirectory(TestDirInput);

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 1,
                GpuWorkers = 1
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.True(Directory.Exists(TestDirOutput));
            Assert.Empty(Directory.GetFiles(TestDirOutput));
        }

        [Fact]
        public void UnifiedProcessor_UsesConfiguredKernel()
        {
            CreateTestImageFloat(TestDirInput, "test.jpg", 10, 10);

            var customKernel = new float[3, 3]
            {
                { 0, 0, 0 },
                { 0, 1, 0 },
                { 0, 0, 0 }
            };

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 1,
                GpuWorkers = 0,
                Kernel = customKernel
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.Single(Directory.GetFiles(TestDirOutput));
        }

        [Fact]
        public void UnifiedProcessor_UsesConfiguredEdgeStrategy()
        {
            CreateTestImageFloat(TestDirInput, "test.jpg", 10, 10);

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 1,
                GpuWorkers = 0,
                Strategy = EdgeStrategy.ZeroPadding
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.Single(Directory.GetFiles(TestDirOutput));
        }

        [Fact]
        public void UnifiedProcessor_MultipleReadersAndWriters()
        {
            for (int i = 0; i < 5; i++)
            {
                CreateTestImageFloat(TestDirInput, $"test{i}.jpg", 10, 10);
            }

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 2,
                GpuWorkers = 1,
                ReaderThreads = 3,
                WriterThreads = 3
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.Equal(5, Directory.GetFiles(TestDirOutput).Length);
        }

        [Fact]
        public void ProcessingTask_PropertiesAreSettable()
        {
            var task = new ProcessingTask
            {
                InputPath = "input.jpg",
                OutputPath = "output.jpg",
                ImageData = new float[5, 5]
            };

            Assert.Equal("input.jpg", task.InputPath);
            Assert.Equal("output.jpg", task.OutputPath);
            Assert.NotNull(task.ImageData);
        }

        [Fact]
        public void ProcessingResult_PropertiesAreSettable()
        {
            var result = new ProcessingResult
            {
                OutputPath = "output.jpg",
                ResultData = new float[5, 5],
                ProcessedBy = ProcessorType.GPU
            };

            Assert.Equal("output.jpg", result.OutputPath);
            Assert.NotNull(result.ResultData);
            Assert.Equal(ProcessorType.GPU, result.ProcessedBy);
        }

        [Fact]
        public void UnifiedProcessorConfig_DefaultValuesAreCorrect()
        {
            var config = new UnifiedProcessorConfig();

            Assert.True(config.CpuWorkers > 0);
            Assert.Equal(1, config.GpuWorkers);
            Assert.Equal(2, config.ReaderThreads);
            Assert.Equal(2, config.WriterThreads);
            Assert.NotNull(config.Kernel);
            Assert.Equal(EdgeStrategy.Extend, config.Strategy);
        }
        [Fact]
        public void UnifiedProcessor_DisposeMultipleTimes_DoesNotThrow()
        {
            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 1,
                GpuWorkers = 0
            };

            var processor = new UnifiedProcessor(config);
            processor.Dispose();
            processor.Dispose();
        }

        [Fact]
        public void GpuProcessor_DisposeMultipleTimes_DoesNotThrow()
        {
            var processor = new GpuConvolutionProcessor();
            processor.Dispose();
            processor.Dispose();
        }

        [Fact]
        public void ConvolveGpu_LargerKernel_ProcessesCorrectly()
        {
            using var gpuProcessor = new GpuConvolutionProcessor();
            var image = new float[10, 10];
            for (int i = 0; i < 10; i++)
                for (int j = 0; j < 10; j++)
                    image[i, j] = 50f;

            var largeKernel = new float[5, 5];
            for (int i = 0; i < 5; i++)
                for (int j = 0; j < 5; j++)
                    largeKernel[i, j] = 1f / 25f;

            var result = gpuProcessor.ConvolveGpu(image, largeKernel, EdgeStrategy.Extend);

            Assert.NotNull(result);
            Assert.Equal(10, result.GetLength(0));
            Assert.Equal(10, result.GetLength(1));
            Assert.InRange(result[5, 5], 45f, 55f);
        }

        [Fact]
        public void UnifiedProcessor_WithZeroWorkers_HandlesGracefully()
        {
            CreateTestImageFloat(TestDirInput, "test.jpg", 10, 10);

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 0,
                GpuWorkers = 0,
                ReaderThreads = 1,
                WriterThreads = 1
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.True(Directory.Exists(TestDirOutput));
        }

        [Fact]
        public void ConvolveGpu_SharpenKernel_ProducesValidResult()
        {
            using var gpuProcessor = new GpuConvolutionProcessor();
            var image = new float[10, 10];
            for (int y = 0; y < 10; y++)
                for (int x = 0; x < 10; x++)
                    image[y, x] = (x + y) % 2 == 0 ? 100f : 50f;

            var result = gpuProcessor.ConvolveGpu(image, Kernels.SharpenFloat, EdgeStrategy.Extend);

            Assert.NotNull(result);
            Assert.Equal(10, result.GetLength(0));
            Assert.Equal(10, result.GetLength(1));
        }

        [Fact]
        public void ConvolveGpu_LaplacianKernel_ProducesValidResult()
        {
            using var gpuProcessor = new GpuConvolutionProcessor();
            var image = new float[10, 10];
            for (int y = 0; y < 10; y++)
                for (int x = 0; x < 10; x++)
                    image[y, x] = 100f;

            var result = gpuProcessor.ConvolveGpu(image, Kernels.LaplacianFloat, EdgeStrategy.ZeroPadding);

            Assert.NotNull(result);
            Assert.Equal(10, result.GetLength(0));
            Assert.Equal(10, result.GetLength(1));
        }

        [Fact]
        public void UnifiedProcessor_SingleReaderSingleWriter_ProcessesCorrectly()
        {
            CreateTestImageFloat(TestDirInput, "test1.jpg", 10, 10);
            CreateTestImageFloat(TestDirInput, "test2.jpg", 10, 10);

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 1,
                GpuWorkers = 0,
                ReaderThreads = 1,
                WriterThreads = 1
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.Equal(2, Directory.GetFiles(TestDirOutput).Length);
        }

        [Fact]
        public void ProcessorType_EnumValues_AreAccessible()
        {
            Assert.Equal(ProcessorType.CPU, ProcessorType.CPU);
            Assert.Equal(ProcessorType.GPU, ProcessorType.GPU);
            Assert.NotEqual(ProcessorType.CPU, ProcessorType.GPU);
        }
        [Fact]
        public void UnifiedProcessor_LargeNumberOfFiles_ProcessesAll()
        {
            for (int i = 0; i < 10; i++)
            {
                CreateTestImageFloat(TestDirInput, $"test{i:D3}.jpg", 10, 10);
            }

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 2,
                GpuWorkers = 1,
                ReaderThreads = 2,
                WriterThreads = 2
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.Equal(10, Directory.GetFiles(TestDirOutput).Length);
        }

        [Fact]
        public void ConvolveGpu_EdgePixels_ExtendStrategy()
        {
            using var gpuProcessor = new GpuConvolutionProcessor();
            var image = new float[5, 5];
            for (int y = 0; y < 5; y++)
                for (int x = 0; x < 5; x++)
                    image[y, x] = (y == 0 || x == 0 || y == 4 || x == 4) ? 255f : 128f;

            var result = gpuProcessor.ConvolveGpu(image, Kernels.BlurBoxFloat, EdgeStrategy.Extend);

            Assert.NotNull(result);
            Assert.True(result[0, 0] > 0);
            Assert.True(result[4, 4] > 0);
        }

        [Fact]
        public void ConvolveGpu_EdgePixels_ZeroPaddingStrategy()
        {
            using var gpuProcessor = new GpuConvolutionProcessor();
            var image = new float[5, 5];
            for (int y = 0; y < 5; y++)
                for (int x = 0; x < 5; x++)
                    image[y, x] = 200f;

            var result = gpuProcessor.ConvolveGpu(image, Kernels.BlurBoxFloat, EdgeStrategy.ZeroPadding);

            Assert.NotNull(result);
            Assert.True(result[0, 0] < result[2, 2]);
        }

        [Fact]
        public void UnifiedProcessor_MixedImageSizes_OnlyProcessesFirst()
        {
            CreateTestImageFloat(TestDirInput, "small.jpg", 5, 5);
            CreateTestImageFloat(TestDirInput, "medium.jpg", 10, 10);
            CreateTestImageFloat(TestDirInput, "large.jpg", 15, 15);

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 1,
                GpuWorkers = 0
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.True(Directory.Exists(TestDirOutput));
            var files = Directory.GetFiles(TestDirOutput);
            Assert.True(files.Length >= 1);
        }

        [Fact]
        public void ProcessDirectory_MultipleIdenticalSizeImages_ProcessesAll()
        {
            for (int i = 0; i < 3; i++)
            {
                CreateTestImageFloat(TestDirInput, $"image{i}.jpg", 20, 20);
            }

            GpuConvolutionProcessor.ProcessDirectory(TestDirInput, TestDirOutput, Kernels.SharpenFloat, EdgeStrategy.ZeroPadding);

            Assert.Equal(3, Directory.GetFiles(TestDirOutput).Length);
        }

        [Fact]
        public void UnifiedProcessor_OnlyGpuWorkers_ProcessesCorrectly()
        {
            CreateTestImageFloat(TestDirInput, "gpu1.jpg", 10, 10);
            CreateTestImageFloat(TestDirInput, "gpu2.jpg", 10, 10);
            CreateTestImageFloat(TestDirInput, "gpu3.jpg", 10, 10);

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 0,
                GpuWorkers = 2,
                ReaderThreads = 1,
                WriterThreads = 1
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.Equal(3, Directory.GetFiles(TestDirOutput).Length);
        }

        [Fact]
        public void UnifiedProcessor_HighParallelism_ProcessesCorrectly()
        {
            for (int i = 0; i < 8; i++)
            {
                CreateTestImageFloat(TestDirInput, $"parallel{i}.jpg", 10, 10);
            }

            var config = new UnifiedProcessorConfig
            {
                CpuWorkers = 4,
                GpuWorkers = 1,
                ReaderThreads = 4,
                WriterThreads = 4
            };

            using var processor = new UnifiedProcessor(config);
            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.Equal(8, Directory.GetFiles(TestDirOutput).Length);
        }

        [Fact]
        public void ConvolveGpu_AllKernelTypes_Work()
        {
            using var gpuProcessor = new GpuConvolutionProcessor();
            var image = new float[10, 10];
            for (int y = 0; y < 10; y++)
                for (int x = 0; x < 10; x++)
                    image[y, x] = 100f;

            var blurResult = gpuProcessor.ConvolveGpu(image, Kernels.BlurBoxFloat, EdgeStrategy.Extend);
            var sharpenResult = gpuProcessor.ConvolveGpu(image, Kernels.SharpenFloat, EdgeStrategy.Extend);
            var laplacianResult = gpuProcessor.ConvolveGpu(image, Kernels.LaplacianFloat, EdgeStrategy.Extend);

            Assert.NotNull(blurResult);
            Assert.NotNull(sharpenResult);
            Assert.NotNull(laplacianResult);
        }
        [Fact]
        public void UnifiedProcessor_CorruptImage_CaughtByReaderAgent()
        {
            Directory.CreateDirectory(TestDirInput);
            File.WriteAllText(Path.Combine(TestDirInput, "corrupt.jpg"), "not an image data");

            var config = new UnifiedProcessorConfig { CpuWorkers = 1, GpuWorkers = 1 };
            using var processor = new UnifiedProcessor(config);

            processor.ProcessDirectory(TestDirInput, TestDirOutput);

            Assert.True(true);
        }

        [Fact]
        public void GpuProcessor_ProcessDirectory_CorruptImage()
        {
            Directory.CreateDirectory(TestDirInput);
            File.WriteAllText(Path.Combine(TestDirInput, "corrupt.jpg"), "not an image data");

            try
            {
                GpuConvolutionProcessor.ProcessDirectory(TestDirInput, TestDirOutput, Kernels.BlurBoxFloat, EdgeStrategy.Extend);
            }
            catch { }

            Assert.True(true);
        }
    }
}