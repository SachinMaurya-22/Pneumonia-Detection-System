
"use strict";

/* =========================================
   PneumoAI - Frontend Interactions
   ========================================= */

document.addEventListener("DOMContentLoaded", () => {
    const input = document.getElementById("xray");
    const dropZone = document.getElementById("dropZone");
    const preview = document.getElementById("imagePreview");
    const previewContainer =
        document.getElementById("previewContainer");
    const fileName = document.getElementById("fileName");
    const removeButton = document.getElementById("removeImage");
    const errorBox = document.getElementById("uploadError");
    const form = document.getElementById("uploadForm");
    const analyzeButton = document.getElementById("analyzeButton");

    // Other pages may not contain an upload form.
    if (!input || !form) {
        return;
    }

    const MAX_FILE_SIZE = 10 * 1024 * 1024;
    const ALLOWED_TYPES = new Set([
        "image/jpeg",
        "image/png"
    ]);

    let previewUrl = null;

    function showError(message) {
        if (!errorBox) {
            return;
        }

        errorBox.textContent = message;
        errorBox.classList.remove("d-none");
    }

    function clearError() {
        if (!errorBox) {
            return;
        }

        errorBox.textContent = "";
        errorBox.classList.add("d-none");
    }

    function clearPreview() {
        if (previewUrl) {
            URL.revokeObjectURL(previewUrl);
            previewUrl = null;
        }

        if (preview) {
            preview.removeAttribute("src");
        }

        if (previewContainer) {
            previewContainer.classList.add("d-none");
        }

        if (fileName) {
            fileName.textContent = "";
        }
    }

    function validateFile(file) {
        if (!file) {
            return "Please select an image first.";
        }

        if (!ALLOWED_TYPES.has(file.type)) {
            return "Only JPG, JPEG and PNG images are supported.";
        }

        if (file.size > MAX_FILE_SIZE) {
            return "The image must be smaller than 10 MB.";
        }

        if (file.size === 0) {
            return "The selected file is empty.";
        }

        return null;
    }

    function processFile(file) {
        clearError();
        clearPreview();

        if (!file) {
            return false;
        }

        const error = validateFile(file);

        if (error) {
            input.value = "";
            showError(error);
            return false;
        }

        if (!preview || !previewContainer) {
            return true;
        }

        previewUrl = URL.createObjectURL(file);
        preview.src = previewUrl;

        if (fileName) {
            fileName.textContent = file.name;
        }

        previewContainer.classList.remove("d-none");
        return true;
    }

    input.addEventListener("change", () => {
        processFile(input.files[0]);
    });

    // Allow clicking the upload area to open the file picker.
    if (dropZone) {
        dropZone.addEventListener("click", (event) => {
            if (event.target !== input) {
                input.click();
            }
        });

        dropZone.addEventListener("keydown", (event) => {
            if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                input.click();
            }
        });

        ["dragenter", "dragover"].forEach((eventName) => {
            dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropZone.classList.add("dragover");
            });
        });

        ["dragleave", "drop"].forEach((eventName) => {
            dropZone.addEventListener(eventName, (event) => {
                event.preventDefault();
                dropZone.classList.remove("dragover");
            });
        });

        dropZone.addEventListener("drop", (event) => {
            const file = event.dataTransfer?.files?.[0];

            if (!file) {
                return;
            }

            try {
                const transfer = new DataTransfer();
                transfer.items.add(file);
                input.files = transfer.files;
                processFile(file);
            } catch {
                clearPreview();
                showError(
                    "Could not attach the dropped file. " +
                    "Please use the file picker instead."
                );
            }
        });
    }

    // Remove selected image.
    if (removeButton) {
        removeButton.addEventListener("click", () => {
            input.value = "";
            clearPreview();
            clearError();
        });
    }

    // Validate before submitting to Flask.
    form.addEventListener("submit", (event) => {
        clearError();

        const file = input.files[0];
        const error = validateFile(file);

        if (error) {
            event.preventDefault();
            showError(error);
            return;
        }

        if (analyzeButton) {
            analyzeButton.disabled = true;
            analyzeButton.setAttribute("aria-busy", "true");
            analyzeButton.textContent = "Uploading image...";
        }
    });

    // Release temporary preview resources.
    window.addEventListener("pagehide", clearPreview);
});
