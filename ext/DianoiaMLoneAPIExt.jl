module DianoiaMLoneAPIExt

using oneAPI: oneAPI, oneAPIBackend
using DianoiaML: DianoiaML, BFloat16

DianoiaML.gpu_backend(::Val{:oneapi}) = oneAPIBackend()

DianoiaML.gpu_precision_available(::oneAPIBackend, ::Type{Float16}) =
    oneAPI.eltype_support().float16

# BFloat16 requires support in both oneArray and the selected SPIR-V compiler.
DianoiaML.gpu_precision_available(::oneAPIBackend, ::Type{BFloat16}) =
    oneAPI.eltype_support().bfloat16 && oneAPI.compiler_config(oneAPI.device()).target.supports_bfloat16

# The graph adapter uses this runtime's internal stream and kernel bindings.
if v"2.10" <= pkgversion(oneAPI) < v"2.11"
    include("replay_oneapi.jl")
end

end
