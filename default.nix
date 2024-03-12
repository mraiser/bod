with (import <nixpkgs> {});
let
  LLP = with pkgs; [
    cargo
    openssl
    pkg-config
    cudatoolkit
    cudaPackages.cudnn
    linuxPackages.nvidia_x11
    dlib
    blas 
    lapack
  ];
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath LLP;
in  
stdenv.mkDerivation {
  name = "udo-env";
  buildInputs = LLP;
  src = null;
  shellHook = ''
    SOURCE_DATE_EPOCH=$(date +%s)
    export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}
    export CUDAToolkit_ROOT=${cudatoolkit.out}:${cudatoolkit.lib}
    export CUDA_ROOT=${cudatoolkit.out}
    export CUDNN_LIB=${cudaPackages.cudnn}
    export CANDLE_NVCC_CCBIN=${gcc11}/bin
    export PATH=${gcc11}/bin:$PATH
  '';
}
