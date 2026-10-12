using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;

using ImageConvolution;

using Xunit;

namespace ImageConvolution.Tests;

public sealed class MeasurementExperimentTests
{
    [Fact]
    public void ByteLayoutReportChecksAllStorageAndPaddingVariants()
    {
        using var workspace = new SmallImageWorkspace();
        string report = Path.Combine(workspace.Root, "byte-layout");
        ByteLayoutExperiment.Run(["--byte-layout", workspace.Input, report, "2"]);

        string summary = File.ReadAllText(Path.Combine(report, "summary.txt"));
        string[] lines = File.ReadAllLines(Path.Combine(report, "times.csv"));
        Assert.Contains("ColumnStorage-Columns: median=", summary);
        Assert.Contains("Columns-P64: median=", summary);
        Assert.Equal(21, lines.Length);
        Assert.All(lines.Skip(1), line => Assert.Equal(3, line.Split(',').Length));
    }

    [Fact]
    public void ByteLayoutHardwareWorkloadRecordsTheChosenVariant()
    {
        using var workspace = new SmallImageWorkspace();
        string report = Path.Combine(workspace.Root, "hardware");
        ByteLayoutExperiment.Run(["--byte-layout", workspace.Input, report, "--hardware", "ColumnStorage-Columns", "2"]);

        using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(report, "workload.json")));
        JsonElement root = document.RootElement;
        Assert.Equal("ColumnStorage-Columns", root.GetProperty("variant").GetString());
        Assert.Equal(2, root.GetProperty("iterations").GetInt32());
        Assert.Equal(9, root.GetProperty("width").GetInt32());
        Assert.Equal(7, root.GetProperty("height").GetInt32());
    }

    [Fact]
    public void ParallelLocalityControlVerifiesOutputsAndWritesFiveComparisons()
    {
        using var workspace = new SmallImageWorkspace();
        string reports = Path.Combine(Environment.CurrentDirectory, "MeasurementResults");
        string[] before = Directory.Exists(reports)
            ? Directory.GetDirectories(reports, "LocalityParallelControl_*")
            : [];
        LocalityExperiment.ParallelControl(workspace.Input, partitions: 2, measuredRuns: 2);
        string folder = Assert.Single(Directory.GetDirectories(reports, "LocalityParallelControl_*").Except(before));

        try
        {
            string summary = File.ReadAllText(Path.Combine(folder, "summary.txt"));
            string[] csv = File.ReadAllLines(Path.Combine(folder, "times.csv"));
            Assert.Contains("Все выходы побитно совпадают", summary);
            Assert.Contains("EqualPartitionsTiledColumns32", summary);
            Assert.Equal(11, csv.Length);
        }
        finally
        {
            Directory.Delete(folder, recursive: true);
        }
    }

    [Theory]
    [InlineData("Rows")]
    [InlineData("Columns")]
    [InlineData("BlockedColumns")]
    [InlineData("ParallelRows")]
    [InlineData("ParallelColumns")]
    public void LocalityHardwareWorkloadRecordsRequestedTraversal(string traversal)
    {
        using var workspace = new SmallImageWorkspace();
        string report = Path.Combine(workspace.Root, traversal + ".json");
        LocalityExperiment.HardwareWorkload(["--locality-profile", traversal, workspace.Input, "2", report]);

        using var document = JsonDocument.Parse(File.ReadAllText(report));
        Assert.Equal(traversal, document.RootElement.GetProperty("variant").GetString());
        Assert.Equal(2, document.RootElement.GetProperty("iterations").GetInt32());
    }

    [Fact]
    public void ExperimentRunnerWritesOneRecordPerMeasuredRun()
    {
        int calls = 0;
        int previousMode = PerformanceProfiler.Mode;
        PerformanceProfiler.Mode = 0;
        string name = "ci-smoke-" + Guid.NewGuid().ToString("N");
        string folder = Path.Combine(ExperimentRunner.ResultsDirectory, name);
        try
        {
            ExperimentRunner.Run(name, () => calls++, warmupRuns: 2);
            string[] rows = File.ReadAllLines(Path.Combine(folder, "times.csv"));
            using var document = JsonDocument.Parse(File.ReadAllText(Path.Combine(folder, "summary.json")));
            Assert.Equal(27, calls);
            Assert.Equal(26, rows.Length);
            Assert.Equal(25, document.RootElement.GetProperty("measuredRuns").GetInt32());
            Assert.Equal(0, document.RootElement.GetProperty("profilerMode").GetInt32());
            Assert.Equal(25, Directory.GetFiles(folder, "run-*.json").Length);
        }
        finally
        {
            PerformanceProfiler.Mode = previousMode;
            if (Directory.Exists(folder)) Directory.Delete(folder, recursive: true);
        }
    }

    private sealed class SmallImageWorkspace : IDisposable
    {
        public string Root { get; } = Path.Combine(Path.GetTempPath(), "imageconv-ci-" + Guid.NewGuid().ToString("N"));
        public string Input => Path.Combine(Root, "input.png");

        public SmallImageWorkspace()
        {
            Directory.CreateDirectory(Root);
            var pixels = new byte[7, 9];
            for (int y = 0; y < 7; y++)
                for (int x = 0; x < 9; x++)
                    pixels[y, x] = (byte)((x * 17 + y * 23) % 256);
            ImageIO.SaveImageByte(pixels, Input);
        }

        public void Dispose() => Directory.Delete(Root, recursive: true);
    }
}