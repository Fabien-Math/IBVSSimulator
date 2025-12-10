{
  description = "Python dev environment with numpy, OpenGL, tqdm, and YAML";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs }: 
    let
      system = "x86_64-linux"; # or "aarch64-linux" if you're on ARM
      pkgs = import nixpkgs { inherit system; };
    in {
      devShells.${system}.default = pkgs.mkShell {
        buildInputs = [
          pkgs.glfw
          pkgs.mesa
          pkgs.libGL
          pkgs.glib
          pkgs.zlib
          pkgs.fontconfig
          pkgs.xorg.libX11
          pkgs.libxkbcommon
          pkgs.freetype
          pkgs.dbus
          pkgs.bashInteractive

          pkgs.python313
          pkgs.python313Packages.scipy
          pkgs.python313Packages.opencv4
          pkgs.python313Packages.osqp
          pkgs.python313Packages.numpy
	        pkgs.python313Packages.seaborn
          pkgs.python313Packages.matplotlib
          pkgs.python313Packages.tqdm
          pkgs.python313Packages.pyopengl
	        pkgs.python313Packages.pyglm
          pkgs.python313Packages.glfw
          pkgs.python313Packages.pyyaml
        ];

        shellHook = ''
          export XDG_SESSION_TYPE=x11
          echo "Python dev environment ready!"
          echo "Packages: numpy, matplotlib, PyOpenGL, glfw, tqdm, PyYAML"
          echo "Python version: $(python --version)"
        '';
      };
    };
}
