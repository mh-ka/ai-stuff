import os

from extractous import Extractor

def from_pdf(file_path: str) -> tuple[str, dict]:

    """
    Extracts text and metadata from a PDF file using the Extractor library.
    """
    
    Args:
        file_path (str): The path to the PDF file.

    Returns:
        tuple[str, dict]: A tuple containing the extracted text and metadata.

    Raises:
        FileNotFoundError: If the file does not exist.

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file '{file_path}' does not exist.")

    extractor = Extractor()
    reader, metadata = extractor.extract_file(file_path)

    result = ""
    buffer = reader.read(4096)
    while buffer:
        result += buffer.decode("utf-8")
        buffer = reader.read(4096)

    return result, metadata

def fix_text_paragraphs(input_file: str, output_file: str) -> None:
    """
    Reads a text file, consolidates lines into paragraphs, and writes the paragraphs to a new file.

    A paragraph is defined as a block of text separated by one or more blank lines.
    Each paragraph in the output file will be a single line of text with spaces between the original lines.

    Parameters:
    input_file (str): The path to the input text file.
    output_file (str): The path to the output text file where the fixed paragraphs will be written.

    Returns:
    None
    """
    # Read all lines from the input file
    try:
        with open(input_file, encoding="utf-8") as file:
            lines: list[str] = file.readlines()
    except FileNotFoundError:
        logger.error(f"Error: The file '{input_file}' was not found.")
        logger.exception("Traceback:")
        return
    except OSError as e:
        logger.error(f"Error reading the file: {e}")
        logger.exception("Traceback:")
        return

    paragraphs: list[str] = []
    current_paragraph: list[str] = []

    # Process each line to form paragraphs
    for line in lines:
        stripped_line: str = line.strip()
        if stripped_line == "":  # Check if the line is empty
            if current_paragraph:
                # Join the lines of the current paragraph with spaces and add to the list of paragraphs
                paragraphs.append(" ".join(current_paragraph).strip())
                current_paragraph = []  # Reset the current paragraph
        else:
            # Add the stripped line to the current paragraph
            current_paragraph.append(stripped_line)

    # Add the last paragraph if there is one
    if current_paragraph:
        paragraphs.append(" ".join(current_paragraph).strip())

    # Write each paragraph to the output file, each on a new line
    with open(output_file, "w", encoding="utf-8") as file:
        for paragraph in paragraphs:
            file.write(paragraph + "\n")
