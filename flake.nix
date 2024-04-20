{
  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/2748d22b45a99fb2deafa5f11c7531c212b2cefa";
  };

  outputs = { self, nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = import nixpkgs {
        inherit system;
        config = {
          allowUnfree = true;
          cudaSupport = true;
          cudaCapabilities = [ "8.6" ];
          cudaEnableForwardCompat = false;
        };
        overlays = [
          # TODO: remove overlay once this is in Cachix:
          # https://github.com/NixOS/nixpkgs/commit/05dedd387f3e8f40f1e9470a1eab51d67aea4ce7
          (final: prev: {
            pythonPackagesExtensions = prev.pythonPackagesExtensions ++ [
              (python-final: python-prev: {
                onnxruntime = (python-prev.onnxruntime.overrideAttrs (old: {
                  buildInputs = old.buildInputs ++ (with pkgs.cudaPackages; [
                    libcublas # libcublasLt.so.XX libcublas.so.XX
                    libcurand # libcurand.so.XX
                    libcufft # libcufft.so.XX
                    cudnn # libcudnn.soXX
                    cuda_cudart # libcudart.so.XX
                  ]);
                }));
              })
            ];
          })
        ];
      };

      python-packages = ps: with ps; [
        numpy
        opencv4
        onnx
        onnxruntime
        scikit-image

        pytest
      ];

      fhs = pkgs.buildFHSUserEnv {
        name = "fhs";
        targetPkgs = pkgs: with pkgs; [
          (python3.withPackages python-packages)
          black
          hatch
        ];
      };
    in
    {
      devShells.${system}.default = fhs.env;
    };
}
