module DianoiaMLMetalExt

using Metal: Metal, MetalBackend
import DianoiaML

DianoiaML.gpu_backend(::Val{:metal}) = MetalBackend()

# The graph adapter uses this runtime's internal compiler and queue bindings.
if v"1.12.1" <= pkgversion(Metal) < v"1.13"
    include("replay_metal.jl")
end

end
