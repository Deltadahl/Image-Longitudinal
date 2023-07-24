with open(r'other\output.toc', 'r') as file:
    content = file.readlines()

for line in content:
    if "\\contentsline" in line:
        parts = line.split("}{")

        # Check if the line has 'numberline' in it
        if "numberline" in parts[1]:
            section_number = parts[1].split("numberline ")[1].replace("{", "").replace("}", " ").strip()
            title = parts[2].split("}")[0].strip()
        else:
            section_number = ""  # or None, or any other default value you prefer
            title = parts[1]

        # Print the section number and title
        print(f"{section_number} {title}".strip())
