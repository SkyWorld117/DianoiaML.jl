module DianoiaMLMetalExt

using Metal: MetalBackend
import DianoiaML

DianoiaML.gpu_backend(::Val{:metal}) = MetalBackend()

end
