{
	description = "DSLR";

	inputs = {
		nixpkgs.url = "github:nixos/nixpkgs?ref=nixpkgs-unstable";
	};

	outputs = { nixpkgs, ... }:
		let
			pkgs = import nixpkgs {
				system = "x86_64-linux";
			};
		in {
			devShells.x86_64-linux = {
				default = pkgs.mkShell {
					packages = [
						pkgs.python3
						pkgs.uv
					];
					shellHook = ''
						source ${pkgs.bash-completion}/share/bash-completion/bash_completion
					'';
				};
				dev = pkgs.mkShell {
					packages = [
						pkgs.nodejs
						pkgs.python3
						pkgs.ruff
						pkgs.uv
					];
					shellHook = ''
						source ${pkgs.bash-completion}/share/bash-completion/bash_completion
					'';
				};
			};
		};
}
