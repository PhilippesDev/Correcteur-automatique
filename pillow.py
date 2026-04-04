from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json

model = ocr_predictor(pretrained=True)

from PIL import Image, ImageEnhance

# Ouvrir l'image
image = Image.open("c:/man.jpg")

enhancer = ImageEnhance.Contrast(image)
image = enhancer.enhance(1.5)
image = image.convert('L')  # Convertir en niveaux de gris

# Sauvegarder l'image modifiée
image.save("c:/test_enhanced.jpg")

document = DocumentFile.from_images("c:/test_enhanced.jpg")
prompt = f"""
Voici un texte brut extrait par un logiciel d'OCR (doctr). 
Il contient probablement des erreurs de lecture, des fautes de frappe ou des caractères mal interprétés.

TA MISSION :
1. Corrige les erreurs de lecture en utilisant le contexte.
2. Reformate le contenu de manière propre (ex: si c'est une facture, fais un tableau ou un JSON).
3. Si un mot est incompréhensible, essaie de le déduire.

TEXTE BRUT :
---
{raw_text}
---
"""

resultat = model(document)

export_json = resultat.export()

with open("exportjson.json", "w") as f : 
    json.dump(export_json, f, indent=4)