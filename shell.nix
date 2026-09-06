{
  pkgs ? import <nixpkgs> { romSupport = true; },
}:

let
  my-python-deps = with pkgs.python3Packages; [
    torchWithRocm
    numpy
    pandas
    matplotlib
    requests
    tiktoken
    pydantic-settings
    tensorflow
    rich
    sympy
    pytest
    pip
  ];
in
pkgs.mkShell {
  name = "Machine Learning Env";
  packages = with pkgs; [
    uv

    jupyter-all
    my-python-deps
    ruff
    black
    mypy
  ];

  venvDir = "/home/artem/projects/ml_learning/ml-venv";

  shellHook = ''
    # Add .wakatime to PATH only if it isn't already there
    if ! echo "$PATH" | tr ':' '\n' | grep -qF "$HOME/.wakatime"; then
      export PATH="$HOME/.wakatime:$PATH"
    fi

     # Create venv if it doesn't exist
     if [ ! -d ".venv" ]; then
       python3 -m venv .venv
     fi
     source .venv/bin/activate

     # Install google-colab-cli if not already installed
    if ! python3 -c "import kaggle" 2>/dev/null; then
       pip install kaggle
     fi

     python3 -c "import torch; x = torch.rand(2, 3); print(x)" && \
     echo "Pytorch successfully installed ✨"
     echo " ---- Environnement Python activé ----✅"
  '';
}
