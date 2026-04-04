
from PIL import ImageDraw
import json


# Ouvrir l'image originale (ou l'image traitée)
image = Image.open(enhanced_path).convert("RGB")
draw = ImageDraw.Draw(image)

# Couleur du soulignement
underline_color = (255, 0, 0)  # rouge
underline_thickness = 3  # épaisseur en pixels

with open("final3.json", "w" ) as result_json :
    final_json = json.load(result_json)

# Parcourir toutes les pages et mots
for page in final_json["pages"]:
    for word in page["words"]:
        (x1, y1), (x2, y2) = word["geometry"]
        # Souligner : ligne horizontale au bas du mot
        draw.line(
            [(x1, y2), (x2, y2)],
            fill=underline_color,
            width=underline_thickness
        )

# Sauvegarder l'image avec soulignements
image.save("final_with_underlines.jpg")
print("✅ Image sauvegardée avec soulignement rouge sur chaque mot.")