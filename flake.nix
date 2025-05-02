{
  description = "A flake providing a dev shell for an experimental project to convert MEG signals to fMRI images.";
  nixConfig = {
    extra-substituters = [
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs =
    { self, nixpkgs }:
    let 
      system = "x86_64-linux"; 
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        config.cudaSupport = true;
      };

      mne-patched = pkgs.python3Packages.mne-python.overridePythonAttrs (old: {
      propagatedBuildInputs = (old.propagatedBuildInputs or []) ++ [ pkgs.python3Packages.setuptools-scm ];
      doCheck = false;
      enableParallelBuilding = false;
    });

    in
    {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = with pkgs; [
          (python3.withPackages (
            ps: with ps; [
              torchWithCuda
              torchvision
              pillow
              matplotlib
              tensorboard
              tqdm
              numpy
              pandas
              nibabel
              scikit-image
              seaborn
              mne-patched
            ]
          ))

        ];

      };
    };
}

