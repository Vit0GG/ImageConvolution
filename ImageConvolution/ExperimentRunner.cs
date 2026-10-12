using System.Diagnostics;
using System.Globalization;
using System.Text;
using System.Text.Json;

namespace ImageConvolution;

public static class ExperimentRunner
{
    public const int MeasuredRuns = 25;
    static string? root;
    public static string ResultsDirectory => root ??= Path.Combine(Environment.CurrentDirectory, "MeasurementResults",
        DateTime.Now.ToString("yyyyMMdd_HHmmss") + "_" + Guid.NewGuid().ToString("N")[..8]);
    static readonly JsonSerializerOptions jsonOptions = new() { WriteIndented = true };
    sealed record RunResult(int Run, double ElapsedMilliseconds, string? Error, PerformanceProfiler.Snapshot? Profile);

    public static void Run(string name, Action action, int warmupRuns = 1)
    {
        string folder = Path.Combine(ResultsDirectory, name);
        Directory.CreateDirectory(folder);
        Console.WriteLine($"\n--- {name}: прогрев {warmupRuns}, измерений {MeasuredRuns} ---");
        for (int i = 0; i < warmupRuns; i++) action();
        var results = new List<RunResult>();
        for (int i = 1; i <= MeasuredRuns; i++)
        {
            using var session = PerformanceProfiler.Mode == 0 ? null : new PerformanceProfiler.Session();
            var sw = Stopwatch.StartNew();
            string? error = null;
            try { action(); }
            catch (Exception ex) { error = ex.ToString(); throw; }
            finally
            {
                sw.Stop();
                var result = new RunResult(i, sw.Elapsed.TotalMilliseconds, error, session?.Capture());
                results.Add(result);
                File.WriteAllText(Path.Combine(folder, $"run-{i:00}.json"), JsonSerializer.Serialize(result, jsonOptions));
                File.WriteAllText(Path.Combine(folder, "times.csv"), "run,elapsed_ms,success\n" +
                    string.Join("\n", results.Select(r => FormattableString.Invariant($"{r.Run},{r.ElapsedMilliseconds:F6},{r.Error == null}"))));
                Console.WriteLine($"Прогон {i}/{MeasuredRuns}: {sw.Elapsed.TotalMilliseconds:F3} мс{(error == null ? "" : " ОШИБКА")}");
            }
        }
        double[] times = results.Select(x => x.ElapsedMilliseconds).ToArray();
        double mean = times.Average();
        var summary = new
        {
            name,
            measuredRuns = MeasuredRuns,
            warmupRuns,
            profilerMode = PerformanceProfiler.Mode,
            meanMilliseconds = mean,
            medianMilliseconds = Median(times),
            standardDeviationPopulationMilliseconds = Math.Sqrt(times.Select(x => (x - mean) * (x - mean)).Average()),
            minMilliseconds = times.Min(),
            maxMilliseconds = times.Max(),
            stageMedians = results.Where(r => r.Profile != null).SelectMany(r => r.Profile!.Stages)
                .GroupBy(s => s.Name).Select(g => new
                {
                    name = g.Key,
                    runsPresent = g.Count(),
                    medianCount = Median(g.Select(s => (double)s.Count)),
                    medianSumMilliseconds = Median(g.Select(s => s.SumMilliseconds))
                }).OrderBy(x => x.name).ToArray()
        };
        File.WriteAllText(Path.Combine(folder, "summary.json"), JsonSerializer.Serialize(summary, jsonOptions));
        var text = new StringBuilder();
        text.AppendLine($"{name}: {MeasuredRuns} измерений, прогревы исключены; профилирование={PerformanceProfiler.Mode}");
        text.AppendLine($"Среднее: {mean:F3} мс; медиана: {summary.medianMilliseconds:F3} мс; σ: {summary.standardDeviationPopulationMilliseconds:F3} мс");
        text.AppendLine($"Минимум: {times.Min():F3} мс; максимум: {times.Max():F3} мс");
        var profiles = results.Where(r => r.Profile != null).Select(r => r.Profile!).ToArray();
        if (profiles.Length > 0)
        {
            text.AppendLine($"Медиана процессорного времени процесса (все потоки), мс: {Median(profiles.Select(p => p.CpuMilliseconds)):F3}");
            text.AppendLine($"Медиана управляемых выделений за прогон, МиБ: {Median(profiles.Select(p => p.AllocatedBytes / (1024.0 * 1024))):F3}");
            text.AppendLine($"Медиана количества сборок GC: Gen0={Median(profiles.Select(p => (double)p.Gen0)):F0}; Gen1={Median(profiles.Select(p => (double)p.Gen1)):F0}; Gen2={Median(profiles.Select(p => (double)p.Gen2)):F0}");
            text.AppendLine("Выделения — накопленный объём, не пиковая память. Процессорное время включает все стадии и служебные потоки процесса.");
        }
        if (summary.stageMedians.Length > 0)
        {
            text.AppendLine("Стадия | медиана количества вызовов | медиана суммы времени за прогон, мс");
            foreach (var stage in summary.stageMedians)
                text.AppendLine($"{stage.name} | {stage.medianCount:F0} | {stage.medianSumMilliseconds:F3}");
            text.AppendLine("Стадии перекрываются, вложенные таймеры учитывают одну работу повторно. Суммы нельзя складывать в общее время.");
            text.AppendLine("Ожидания нескольких агентов суммируются и могут превышать время всего прогона.");
            text.AppendLine("Таймеры стадий измеряют прошедшее время вызовов, включая вытеснение потоков; это не чистое процессорное время.");
            text.AppendLine("wait_input/enqueue: длительность операции очереди вместе с ожиданием, не отдельный замер блокировки.");
            text.AppendLine("Чтение включает декодирование и может обслуживаться файловым кэшем; сохранение не гарантирует физическую запись на носитель к моменту остановки таймера.");
        }
        File.WriteAllText(Path.Combine(folder, "summary.txt"), text.ToString());
        Console.WriteLine(text.ToString());
        Console.WriteLine($"Результаты: {folder}");
    }

    static double Median(IEnumerable<double> values)
    {
        var sorted = values.OrderBy(x => x).ToArray();
        return sorted.Length % 2 == 1 ? sorted[sorted.Length / 2] : (sorted[sorted.Length / 2 - 1] + sorted[sorted.Length / 2]) / 2;
    }
}