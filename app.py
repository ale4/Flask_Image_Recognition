"""Handle image uploads and display hand-digit predictions."""

from flask import Flask, render_template, request
from model import preprocess_img, predict_result

app = Flask(__name__)


@app.route("/")
def main():
    """Display the image-upload page."""
    return render_template("index.html")


@app.route("/prediction", methods=["POST"])
def predict_image_file():
    """Process an uploaded image and display its predicted digit."""
    try:
        img = preprocess_img(request.files["file"].stream)
        pred = predict_result(img)
        return render_template("result.html", predictions=str(pred))
    # Preserve the original application's catch-all error response.
    except:  # pylint: disable=bare-except
        return render_template(
            "result.html",
            err="File cannot be processed.",
        )


if __name__ == "__main__":
    app.run(port=9000, debug=True)
