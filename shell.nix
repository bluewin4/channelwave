{ pkgs ? import <nixpkgs> {} }:

pkgs.mkShell {
  buildInputs = [
    pkgs.python312
    pkgs.uv
  ];

  shellHook = ''
    uv venv --allow-existing
    # Only install if not already installed (check by trying to import)
    if ! "$PWD/.venv/bin/python" -c "import pufferlib" 2>/dev/null; then
      uv pip install -e .
    fi
    source .venv/bin/activate
  '';
}
