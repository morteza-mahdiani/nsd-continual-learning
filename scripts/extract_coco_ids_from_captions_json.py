import json

# Load your JSON file
input_file = '/home/mahdiani/projects/def-charesti/mahdiani/data/mpnet_embeddings/captions_metadata.json'  # change to your actual file name

# Read the JSON content
with open(input_file, 'r') as f:
    data = json.load(f)

# Extract COCO IDs (as strings or ints as needed)
coco_ids = list(data.keys())

# Optional: convert to integers
# coco_ids = [int(cid) for cid in coco_ids]

# Print the list
print("Extracted COCO IDs:")
print(coco_ids)

# Save to file
# Option 1: Save as JSON list
with open('./data/coco_ids.json', 'w') as f_json:
    json.dump(coco_ids, f_json)

# # Option 2: Save as plain text (one ID per line)
# with open('data/coco_meta/coco_ids.txt', 'w') as f_txt:
#     for cid in coco_ids:
#         f_txt.write(f"{cid}\n")

print("COCO IDs saved to 'coco_ids.json' and 'coco_ids.txt'.")
