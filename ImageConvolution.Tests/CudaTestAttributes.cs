using System;

using ImageConvolution;

using Xunit;

namespace ImageConvolution.Tests;

internal static class CudaAvailability
{
    private static readonly Lazy<bool> available = new(() =>
    {
        try { return GpuConvolutionProcessor.GetDeviceCount() > 0; }
        catch (Exception) { return false; }
    });

    public static bool IsAvailable => Environment.GetEnvironmentVariable("IMAGECONV_SKIP_CUDA_TESTS") != "1"
        && available.Value;
}

public sealed class CudaFactAttribute : FactAttribute
{
    public CudaFactAttribute()
    {
        if (!CudaAvailability.IsAvailable) Skip = "Requires a CUDA GPU.";
    }
}

public sealed class CudaTheoryAttribute : TheoryAttribute
{
    public CudaTheoryAttribute()
    {
        if (!CudaAvailability.IsAvailable) Skip = "Requires a CUDA GPU.";
    }
}