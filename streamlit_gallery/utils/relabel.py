from streamlit_gallery.utils.labels import get_labels

def relabel(predicted : int) -> str:
    labels = get_labels()

    return labels[predicted]