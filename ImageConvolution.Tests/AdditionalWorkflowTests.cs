using System;
using System.IO;
using System.Linq;
using System.Text.Json;

using ImageConvolution;

using Xunit;

namespace ImageConvolution.Tests;

public sealed class AdditionalWorkflowTests
{
    [Fact]
    public void SequentialFolderProcessingWritesTheExpectedPixels()
    {
        using var workspace = new ImageWorkspace();
        string output = Path.Combine(workspace.Root, "sequential");

        BatchProcessor.ProcessImagesSequential(workspace.InputDirectory, output);

        byte[,] actual = ImageIO.LoadAsGrayscaleByte(Path.Combine(output, "test.png"));
        byte[,] expected = ConvolutionProcessor.Convolve(workspace.Pixels, Kernels.BlurBox, EdgeStrategy.Extend);
        AssertPixelsEqual(expected, actual);
    }

    [Fact]
    public void DirectImageSharpFolderProcessingWritesAnImageAndProfilesItsStages()
    {
        using var workspace = new ImageWorkspace();
        string output = Path.Combine(workspace.Root, "direct");
        using var session = new PerformanceProfiler.Session();

        LibraryProcessor.ProcessImagesWithImageSharpDirect(workspace.InputDirectory, output);

        var profile = session.Capture();
        Assert.True(File.Exists(Path.Combine(output, "test.png")));
        byte[,] result = ImageIO.LoadAsGrayscaleByte(Path.Combine(output, "test.png"));
        Assert.Equal(workspace.Pixels.GetLength(0), result.GetLength(0));
        Assert.Equal(workspace.Pixels.GetLength(1), result.GetLength(1));
        Assert.Equal(1L, Assert.Single(profile.Stages, stage => stage.Name == "imagesharp.direct.blur").Count);
    }

    [Fact]
    public void MissingInputFolderDoesNotCreateAnOutputFolder()
    {
        using var workspace = new ImageWorkspace();
        string output = Path.Combine(workspace.Root, "missing-output");
        BatchProcessor.ProcessImagesSequential(Path.Combine(workspace.Root, "missing"), output);
        LibraryProcessor.ProcessImagesWithImageSharpDirect(Path.Combine(workspace.Root, "missing"), output);
        Assert.False(Directory.Exists(output));
    }

    [Fact]
    public void LocalityRunRecordsEveryTraversal()
    {
        using var workspace = new ImageWorkspace();
        string results = ExperimentRunner.ResultsDirectory;
        string[] names = ["Locality-Rows", "Locality-Columns", "Locality-BlockedColumns"];
        try
        {
            LocalityExperiment.Run(workspace.InputPath);
            foreach (string name in names)
            {
                string folder = Path.Combine(results, name);
                using var summary = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "summary.json")));
                Assert.Equal(25, summary.RootElement.GetProperty("measuredRuns").GetInt32());
            }
        }
        finally
        {
            foreach (string name in names)
            {
                string folder = Path.Combine(results, name);
                if (Directory.Exists(folder)) Directory.Delete(folder, recursive: true);
            }
        }
    }

    [Fact]
    public void ProfilingExperimentRecordsStageSummary()
    {
        int previousMode = PerformanceProfiler.Mode;
        string name = "profile-smoke-" + Guid.NewGuid().ToString("N");
        string folder = Path.Combine(ExperimentRunner.ResultsDirectory, name);
        try
        {
            PerformanceProfiler.Mode = 1;
            ExperimentRunner.Run(name, () => PerformanceProfiler.Measure("test.stage", () => { }), warmupRuns: 0);
            using var summary = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "summary.json")));
            JsonElement stage = Assert.Single(summary.RootElement.GetProperty("stageMedians").EnumerateArray());
            Assert.Equal("test.stage", stage.GetProperty("name").GetString());
            Assert.Equal(25, stage.GetProperty("runsPresent").GetInt32());
            Assert.Contains("Медиана процессорного времени", File.ReadAllText(Path.Combine(folder, "summary.txt")));
        }
        finally
        {
            PerformanceProfiler.Mode = previousMode;
            if (Directory.Exists(folder)) Directory.Delete(folder, recursive: true);
        }
    }

    [Fact]
    public void DescribeArrayCapturesElementSizeAndAddressSteps()
    {
        using var session = new PerformanceProfiler.Session();
        PerformanceProfiler.DescribeArray(new byte[2, 3]);
        var metadata = session.Capture().Metadata;
        Assert.Contains("layout.Byte.3x2", metadata.Keys);
    }

    [Fact]
    public void InvalidConvolutionInputsAreRejected()
    {
        byte[,] pixels = new byte[2, 2];
        Assert.Throws<ArgumentNullException>(() => ConvolutionProcessor.Convolve(null!, Kernels.BlurBox));
        Assert.Throws<ArgumentNullException>(() => ConvolutionProcessor.Convolve(pixels, null!));
        Assert.Throws<ArgumentException>(() => ConvolutionProcessor.Convolve(new byte[0, 2], Kernels.BlurBox));
        Assert.Throws<ArgumentException>(() => ConvolutionProcessor.Convolve(pixels, new float[,] { { float.NaN } }));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            ConvolutionProcessor.Convolve(pixels, Kernels.BlurBox, (EdgeStrategy)123));
    }

    [Fact]
    public void ParallelColumnFolderProcessingMatchesSequentialPixels()
    {
        using var workspace = new ImageWorkspace();
        string output = Path.Combine(workspace.Root, "columns");
        BatchProcessor.ProcessImagesNaiveParallel(workspace.InputDirectory, output, ParallelStrategy.ParallelByColumns, 2);
        byte[,] actual = ImageIO.LoadAsGrayscaleByte(Path.Combine(output, "test.png"));
        byte[,] expected = ConvolutionProcessor.Convolve(workspace.Pixels, Kernels.BlurBox, EdgeStrategy.Extend);
        AssertPixelsEqual(expected, actual);
    }

    [Fact]
    public void FolderProcessorsRejectUsingTheInputAsOutput()
    {
        using var workspace = new ImageWorkspace();
        Assert.Throws<ArgumentException>(() =>
            BatchProcessor.ProcessImagesSequential(workspace.InputDirectory, workspace.InputDirectory));
        Assert.Throws<ArgumentException>(() =>
            LibraryProcessor.ProcessImagesWithImageSharpDirect(workspace.InputDirectory, workspace.InputDirectory));
    }

    [Fact]
    public void InvalidPipelineConfigurationFailsBeforeStartingWorkers()
    {
        Assert.Throws<ArgumentException>(() => new UnifiedProcessor(new UnifiedProcessorConfig
        {
            CpuWorkers = 1,
            GpuWorkers = 0,
            ReaderThreads = 0
        }));
        Assert.Throws<ArgumentOutOfRangeException>(() => new UnifiedProcessor(new UnifiedProcessorConfig
        {
            CpuWorkers = 1,
            GpuWorkers = 0,
            CpuStrategy = (ParallelStrategy)123
        }));
    }

    [Fact]
    public void InvalidLocalityOptionsAreRejected()
    {
        byte[,] pixels = new byte[2, 2];
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LocalityExperiment.Convolve(pixels, Kernels.BlurBox, EdgeStrategy.Extend, (SequentialTraversal)123));
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            LocalityExperiment.Convolve(pixels, Kernels.BlurBox, EdgeStrategy.Extend, SequentialTraversal.Rows, blockSize: 0));
        Assert.Throws<ArgumentException>(() => LocalityExperiment.HardwareWorkload(["--locality-profile"]));
    }

    private static void AssertPixelsEqual(byte[,] expected, byte[,] actual)
    {
        Assert.Equal(expected.GetLength(0), actual.GetLength(0));
        Assert.Equal(expected.GetLength(1), actual.GetLength(1));
        for (int y = 0; y < expected.GetLength(0); y++)
            for (int x = 0; x < expected.GetLength(1); x++) Assert.Equal(expected[y, x], actual[y, x]);
    }

    private sealed class ImageWorkspace : IDisposable
    {
        public string Root { get; } = Path.Combine(Path.GetTempPath(), "imageconv-workflow-" + Guid.NewGuid().ToString("N"));
        public string InputDirectory => Path.Combine(Root, "input");
        public string InputPath => Path.Combine(InputDirectory, "test.png");
        public byte[,] Pixels { get; } = new byte[7, 9];

        public ImageWorkspace()
        {
            Directory.CreateDirectory(InputDirectory);
            for (int y = 0; y < Pixels.GetLength(0); y++)
                for (int x = 0; x < Pixels.GetLength(1); x++) Pixels[y, x] = (byte)((x * 17 + y * 23) % 256);
            ImageIO.SaveImageByte(Pixels, InputPath);
        }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}