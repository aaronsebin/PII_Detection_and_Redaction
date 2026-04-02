from server import app, main as _main

__all__ = ["app", "main"]


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    _main(host=host, port=port)


if __name__ == "__main__":
    main()
