{
  pkgs,
  ...
}:

{
  # https://devenv.sh/packages/
  packages = with pkgs; [
    # Debug tools
    minicom
    gcc

    # Python stuff
    ruff
  ];

  # https://devenv.sh/languages/
  languages = {
    cplusplus = {
      enable = true;
      lsp = {
        enable = true;
        package = pkgs.clang-tools;
      };
    };
    python = {
      enable = true;
      version = "3.12";
      venv.enable = true;
      uv = {
        enable = true;
        sync.enable = true;
      };
      lsp = {
        enable = true;
        package = pkgs.ty;
      };
    };
  };
}
