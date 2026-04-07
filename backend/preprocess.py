def preprocess_image(image_path: str):
    import cv2
    from PIL import Image

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return Image.fromarray(gray).convert("RGB")