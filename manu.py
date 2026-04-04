from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json


# 1. On change pour un modèle de reconnaissance plus performant (ViT)
model = ocr_predictor(det_arch='db_resnet50', reco_arch='vitstr', pretrained=True)

#fine tuning pillow niveau de contraste et luminosité
from PIL import Image, ImageEnhance
# Ouvrir l'image
image = Image.open("c:/man.jpg") 

# Augmenter le contraste
enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.5)

# Sauvegarder l'image modifiée
image.save("c:/man_enhanced.jpg")

# 3. Analyser avec un seuil de détection plus bas si le manuscrit est fin
document = DocumentFile.from_images("c:/man_enhanced.jpg")
resultat = model(document)

# 4. Afficher le résultat visuellement pour voir où il se trompe
# (Cela va ouvrir une fenêtre avec les boîtes tracées)
json_output = resultat.export()

with open("output5.json", "w") as f:
    json.dump(json_output, f, indent=4)

# 5. Export
