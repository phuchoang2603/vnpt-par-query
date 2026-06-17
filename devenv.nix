{
  pkgs,
  ...
}:

{
  # https://devenv.sh/packages/
  packages = with pkgs; [
    gcc

    minicom
  ];

  # https://devenv.sh/languages/
  languages.cplusplus = {
    enable = true;
    lsp = {
      enable = true;
      package = pkgs.clang-tools;
    };
  };
}
