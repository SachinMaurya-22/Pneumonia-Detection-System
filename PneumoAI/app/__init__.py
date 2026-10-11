
"""PneumoAI Flask application factory."""

import os

from flask import Flask, render_template
from app.routes import register_routes


def create_app():
    app = Flask(__name__)

    # Upload limit: 10 MB
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    # Configure a strong secret through environment variables.
    secret_key = os.environ.get("SECRET_KEY")

    if not secret_key:
        raise RuntimeError(
            "SECRET_KEY environment variable is not configured."
        )

    app.config["SECRET_KEY"] = secret_key

    # Register application routes.
    register_routes(app)

    # Custom 404 page.
    @app.errorhandler(404)
    def page_not_found(error):
        return render_template("404.html"), 404

    # Custom upload-size error.
    @app.errorhandler(413)
    def request_entity_too_large(error):
        return (
            "Uploaded file exceeds the 10 MB size limit.",
            413,
        )

    return app
