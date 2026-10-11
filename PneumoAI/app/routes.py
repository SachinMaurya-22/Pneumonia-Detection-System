
"""Routes for the PneumoAI web application."""

import logging

from flask import (
    current_app,
    flash,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from app.model_service import predict_image
from app.utils import ImageValidationError, validate_image


logger = logging.getLogger(__name__)


def register_routes(app):
    """Register all application endpoints."""

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/predict", methods=["POST"])
    def predict():
        uploaded_file = request.files.get("file")

        if uploaded_file is None or not uploaded_file.filename:
            flash("Please select a chest X-ray image.", "warning")
            return redirect(url_for("index"))

        filename = secure_filename(uploaded_file.filename)

        if not filename:
            flash("Invalid filename.", "danger")
            return redirect(url_for("index"))

        try:
            # Read into memory; the original upload is not saved.
            file_bytes = uploaded_file.read()

            validate_image(file_bytes)

            result = predict_image(file_bytes)

            return render_template(
                "result.html",
                prediction=result["prediction"],
                confidence=result["confidence"],
                image_filename=filename,
            )

        except ImageValidationError as exc:
            flash(str(exc), "danger")
            return redirect(url_for("index"))

        except FileNotFoundError:
            logger.exception("Trained model file is missing.")
            flash(
                "The trained model is not configured. "
                "Please contact the project administrator.",
                "danger",
            )
            return redirect(url_for("index"))

        except Exception:
            # Keep internal errors out of the browser response.
            logger.exception("Image prediction failed.")
            flash(
                "Image analysis failed. Check the model configuration "
                "and server logs, then try again.",
                "danger",
            )
            return redirect(url_for("index"))
