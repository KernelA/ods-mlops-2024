import io

from IPython.display import Image


def to_iimage(figure):
    figure.tight_layout()
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png")
    return Image(data=buffer.getvalue(), format="png")
