module DianoiaMLAMDGPUExt

using AMDGPU: AMDGPU, ROCBackend
import DianoiaML
using LinearAlgebra: mul!

DianoiaML.gpu_backend(::Val{:amd}) = ROCBackend()

include("replay_rocm.jl")

function DianoiaML.gpu_convolution_multiply!(::ROCBackend, output, left, right, α::Float32)
    m, n = size(output)
    k = size(left, 2)
    if eltype(left) === eltype(right) === eltype(output) === Float32 &&
       min(m, n, k) >= 32 && m * n * k >= 2^24 && AMDGPU.functional(:rocblas)
        return mul!(output, left, right, α, 0f0)
    end
    return DianoiaML.gpu_multiply!(output, left, right, Float32, α)
end

end
