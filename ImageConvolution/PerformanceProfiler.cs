using System.Collections.Concurrent;
using System.Diagnostics;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Threading.Channels;

namespace ImageConvolution;

public static class PerformanceProfiler
{
    static readonly AsyncLocal<Session?> current = new();
    public static int Mode { get; set; } = 1;
    public static bool IsActive => current.Value != null;
    public static bool DetailedGpu => current.Value != null && Mode == 2;

    public sealed record Stage(string Name, long Count, double SumMilliseconds, double MeanMilliseconds);
    public sealed record Snapshot(double CpuMilliseconds, long AllocatedBytes, int Gen0, int Gen1, int Gen2,
        Dictionary<string, object> Metadata, Stage[] Stages);

    public sealed class Session : IDisposable
    {
        internal sealed class Counter { public long Ticks; public long Count; }
        readonly ConcurrentDictionary<string, Counter> counters = new();
        readonly ConcurrentDictionary<string, object> metadata = new();
        readonly Session? previous;
        readonly long allocated = GC.GetTotalAllocatedBytes(false);
        readonly int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);
        readonly Process process = Process.GetCurrentProcess();
        readonly TimeSpan cpuStart;

        public Session()
        {
            cpuStart = process.TotalProcessorTime;
            previous = current.Value;
            current.Value = this;
            metadata["profilingMode"] = Mode;
            metadata["logicalProcessors"] = Environment.ProcessorCount;
            metadata["runtime"] = RuntimeInformation.FrameworkDescription;
            metadata["os"] = RuntimeInformation.OSDescription;
        }

        internal void Add(string name, long ticks)
        {
            var counter = counters.GetOrAdd(name, _ => new());
            Interlocked.Add(ref counter.Ticks, ticks);
            Interlocked.Increment(ref counter.Count);
        }

        internal void Set(string key, object value) => metadata[key] = value;

        public Snapshot Capture() => new((process.TotalProcessorTime - cpuStart).TotalMilliseconds,
            GC.GetTotalAllocatedBytes(false) - allocated, GC.CollectionCount(0) - g0,
            GC.CollectionCount(1) - g1, GC.CollectionCount(2) - g2, metadata.ToDictionary(x => x.Key, x => x.Value),
            counters.OrderBy(x => x.Key).Select(x => new Stage(x.Key, x.Value.Count,
                x.Value.Ticks * 1000.0 / Stopwatch.Frequency,
                x.Value.Count == 0 ? 0 : x.Value.Ticks * 1000.0 / Stopwatch.Frequency / x.Value.Count)).ToArray());

        public void Dispose()
        {
            current.Value = previous;
            process.Dispose();
        }
    }

    public readonly struct Scope : IDisposable
    {
        readonly Session? session;
        readonly string name;
        readonly long start;
        internal Scope(string name)
        {
            session = current.Value;
            this.name = name;
            start = session == null ? 0 : Stopwatch.GetTimestamp();
        }
        public void Dispose()
        {
            if (session != null) session.Add(name, Stopwatch.GetTimestamp() - start);
        }
    }

    public static Scope Measure(string name) => new(name);
    public static T Measure<T>(string name, Func<T> action)
    {
        using var scope = Measure(name);
        return action();
    }
    public static void Measure(string name, Action action)
    {
        using var scope = Measure(name);
        action();
    }
    public static void Set(string name, object value) => current.Value?.Set(name, value);
    public static void AddMilliseconds(string name, double milliseconds) => current.Value?.Add(name,
        (long)(milliseconds * Stopwatch.Frequency / 1000.0));

    public static void DescribeArray<T>(T[,] data) where T : unmanaged
    {
        if (current.Value == null || data.Length == 0) return;
        int height = data.GetLength(0), width = data.GetLength(1);
        Set($"layout.{typeof(T).Name}.{width}x{height}", new
        {
            width,
            height,
            elementBytes = Unsafe.SizeOf<T>(),
            nextColumnBytes = width > 1 ? (long?)Unsafe.ByteOffset(ref data[0, 0], ref data[0, 1]) : null,
            nextRowBytes = height > 1 ? (long?)Unsafe.ByteOffset(ref data[0, 0], ref data[1, 0]) : null,
            indexing = "[y,x]; nextColumn = horizontal; nextRow = vertical"
        });
    }

    public static IEnumerable<T> Consume<T>(BlockingCollection<T> queue, string stage, CancellationToken token)
    {
        using var iterator = queue.GetConsumingEnumerable(token).GetEnumerator();
        while (true)
        {
            bool available;
            using (Measure(stage)) available = iterator.MoveNext();
            if (!available) yield break;
            yield return iterator.Current;
        }
    }

    public static async IAsyncEnumerable<T> ConsumeAsync<T>(ChannelReader<T> reader, string stage,
        [EnumeratorCancellation] CancellationToken token)
    {
        await using var iterator = reader.ReadAllAsync(token).GetAsyncEnumerator(token);
        while (true)
        {
            bool available;
            using (Measure(stage)) available = await iterator.MoveNextAsync();
            if (!available) yield break;
            yield return iterator.Current;
        }
    }
}