"""
Named Entity Recognition (NER) using spaCy
Course: Natural Language Processing (ET5M004)
"""

import spacy
from spacy import displacy
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Pre-trained spaCy Model
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# 2. Sample Text for Demonstration
# -----------------------------
sample_text = """
Elon Musk founded SpaceX in California in 2002. 
Later, he became the CEO of Tesla and acquired Twitter for $44 billion in October 2022. 
Apple Inc., headquartered in Cupertino, was founded by Steve Jobs in 1976. 
Barack Obama served as the President of the United States from 2009 to 2017.
"""

print("===== Original Text =====")
print(sample_text)

# -----------------------------
# 3. Extract Named Entities
# -----------------------------
doc = nlp(sample_text)

print("\n===== Extracted Entities =====")
entities_data = []
for ent in doc.ents:
    print(f"{ent.text:<30} -> {ent.label_:<10} ({spacy.explain(ent.label_)})")
    entities_data.append({'Entity': ent.text, 'Label': ent.label_, 'Description': spacy.explain(ent.label_)})

# -----------------------------
# 4. Save Entities to CSV
# -----------------------------
entities_df = pd.DataFrame(entities_data)
entities_df.to_csv("../output/extracted_entities.csv", index=False)
print("\nEntities saved to output/extracted_entities.csv")

# -----------------------------
# 5. Visualize Entity Type Distribution
# -----------------------------
label_counts = entities_df['Label'].value_counts()

plt.figure(figsize=(8, 5))
label_counts.plot(kind='bar', color='teal')
plt.title('Entity Type Distribution')
plt.xlabel('Entity Type')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../output/entity_distribution.png")
plt.show()

# -----------------------------
# 6. Generate Visual HTML Highlighting (saved as file)
# -----------------------------
html = displacy.render(doc, style="ent", page=True)
with open("../output/entity_visualization.html", "w", encoding="utf-8") as f:
    f.write(html)
print("Visual entity highlighting saved to output/entity_visualization.html (open in browser to view)")

# -----------------------------
# 7. Interactive / Live Check
# -----------------------------
print("\n===== Live NER Checker =====")
while True:
    user_input = input("\nEnter a sentence to extract entities (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("Exiting... Thank you!")
        break
    
    doc_live = nlp(user_input)
    
    if len(doc_live.ents) == 0:
        print("No named entities found in this sentence.")
    else:
        print("Entities found:")
        for ent in doc_live.ents:
            print(f"  - {ent.text} ({ent.label_}: {spacy.explain(ent.label_)})")