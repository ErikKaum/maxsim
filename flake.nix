{
  inputs = {
    # Pinned to the same kernel-builder rev that kernels-community currently
    # ships against (verified by reading flake.lock on kernels-community/
    # activation). Avoids the triton-3.7.0-cp313 hash mismatch that the
    # latest kernel-builder pulls in via its newer nixpkgs pin.
    kernel-builder.url = "github:huggingface/kernels/614c6bb2ee922a832852cb5562d5571cd9600a9c";
  };
  outputs =
    { self, kernel-builder, ... }:
    kernel-builder.lib.genKernelFlakeOutputs {
      inherit self;
      path = ./.;

      # Extra Python packages available inside `kernel-builder devshell` /
      # `testshell` (and in the corresponding `nix develop` shells). These
      # are not bundled into the kernel uploaded to the Hub — they only
      # exist for local development and tests.
      pythonCheckInputs = pkgs: with pkgs; [ pytest numpy kernels tabulate matplotlib ];
    };
}
