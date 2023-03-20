from pathlib import Path
import matplotlib.pyplot as plt
import quickdraw as qd


image_size = (100, 100)

# Fonction qui récupère les images dans la base de données de Google
def generate_class_images(name, max_drawings, recognized):
    directory = Path('./dataset/' + name)

    if not directory.exists():
        directory.mkdir(parents=True)

# Recognized=True = les images reconnues par l'IA de Google
    images = qd.QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized)
    for img in images.drawings:
        filename = directory.as_posix() + "/" + str(img.key_id) + ".png"
        img.get_image(stroke_width=3).resize(image_size).save(filename)

for label in qd.QuickDrawData().drawing_names:
    generate_class_images(label, max_drawings=1200, recognized=True)
