
"""
PneumoAI - Image Utilities

Responsibilities:
1. Validate uploaded image bytes.
2. Restrict supported image formats and file size.
3. Detect invalid or corrupted images.
4. Preprocess images for the trained Keras model.
"""

from io import BytesIO

import numpy as np
from PIL import (
    Image,
    ImageOps,
    UnidentifiedImageError,
)


# Configuration
ALLOWED_FORMATS = {"JPEG", "PNG"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

# Must match your model's training configuration.
DEFAULT_IMAGE_SIZE = (224, 224)


class ImageValidationError(ValueError):
    """Raised when an uploaded image is invalid."""


def validate_image(file_bytes: bytes) -> None:
    """
    Validate uploaded image content.

    Checks:
    - File is not empty.
    - File size is within the configured limit.
    - Actual image format is JPEG or PNG.
    - Image can be decoded.
    """

    if not file_bytes:
        raise ImageValidationError(
            "Please select an image."
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise ImageValidationError(
            "Image must be smaller than 10 MB."
        )

    try:
        # Verify the image file structure.
        with Image.open(BytesIO(file_bytes)) as image:

            if image.format not in ALLOWED_FORMATS:
                raise ImageValidationError(
                    "Only JPG, JPEG and PNG images are supported."
                )

            if image.width < 1 or image.height < 1:
                raise ImageValidationError(
                    "Invalid image dimensions."
                )

            image.verify()

        # Verify that the image can actually be decoded.
        with Image.open(BytesIO(file_bytes)) as image:
            image.load()

    except ImageValidationError:
        raise

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise ImageValidationError(
            "Invalid or corrupted image. Please upload another image."
        ) from exc


def preprocess_image(
    file_bytes: bytes,
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> np.ndarray:
    """
    Convert an uploaded image into a model-ready NumPy batch.

    Returns:
        Array with shape (1, height, width, 3).

    Important:
        Pixel values are normalized to [0, 1].
        This must match the model's expected input pipeline.
    """

    validate_image(file_bytes)

    try:
        with Image.open(BytesIO(file_bytes)) as image:

            # Respect image orientation metadata.
            image = ImageOps.exif_transpose(image)

            # Convert grayscale or other supported modes to RGB.
            image = image.convert("RGB")

            # Resize to the configured model input dimensions.
            image = image.resize(
                image_size,
                Image.Resampling.BILINEAR,
            )

            # Convert to floating-point pixels.
            array = np.asarray(
                image,
                dtype=np.float32,
            )

            # Normalize pixel values to [0, 1].
            array = array / 255.0

            # Add batch dimension.
            batch = np.expand_dims(array, axis=0)

            return batch

    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
    ) as exc:
        raise ImageValidationError(
            "Unable to preprocess the uploaded image."
        ) from exc
