import os
from os import path as p

def change_ai_context(file: str, content: str) -> None:
    if not p.exists(file):
        raise FileExistsError(f"File '{file}' does not exist.")

    with open(file, 'a') as f:
        f.write(content + '\n')  # Ensure new content is added properly

# Example usage
file_path = "AI_context.txtERROR"
new_content = "Appending this line to AI_context.txt"
change_ai_context(file_path, new_content)