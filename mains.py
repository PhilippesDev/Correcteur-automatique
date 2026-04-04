from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import json

#initialisation du modele
model = ocr_predictor(pretrained=True)

#preparation du document
document = DocumentFile.from_images("c:/man.jpg")

#extraction du texte
resultat = model(document)

json_out = resultat.export()

with open("output3.json", "w") as fichier_de_sortie :
    json.dump(json_out, fichier_de_sortie, indent=4)

print("LE TEXTE A ETE EXPORTE AVEC SUCCEES ")