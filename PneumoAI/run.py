
import os

from app import create_app


def main():
    """Create and run the Flask application."""

    app = create_app()

    debug_mode = os.getenv("FLASK_DEBUG", "0") == "1"

    app.run(
        host="127.0.0.1",
        port=int(os.getenv("PORT", "5000")),
        debug=debug_mode,
    )


if __name__ == "__main__":
    main()
