from pathlib import Path

class FileUtils:
    """Utility class for file operations."""
    @staticmethod
    def delete_all_files(path: str | Path,
                         except_git_keep:bool = True) -> None:
        path = Path(path)

        if not path.is_dir():
            raise ValueError(f"{path} is not a valid directory")

        for file in path.iterdir():
            if except_git_keep:
                if file.is_file() and file.name != ".gitkeep":
                    file.unlink()
            else:
                if file.is_file():
                    file.unlink()
