import os
from dotenv import load_dotenv
import nest_asyncio
import re
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
from concurrent.futures import ThreadPoolExecutor
import logging

load_dotenv()
nest_asyncio.apply()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
BATCH_SIZE = 5  # Number of documents to process in parallel

def get_files_in_directory(path):
    """Gets all files in the specified directory.

    Args:
        path: The directory path.

    Returns:
        A list of filenames in the directory.
    """

    return [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)))]
    # Modify the above accordingly if you want to ignore some kind of file formats
    # return [f for f in os.listdir(path) if (os.path.isfile(os.path.join(path, f)) and f[-5:]!="ipynb" and f[-3:]!="png")]

def process_document(file_info):
    """Process a single document with error handling."""
    input_path, output_path, file_extension = file_info
    try:
        parser = LlamaParse(result_type="markdown")
        file_extractor = {file_extension: parser}
        documents = SimpleDirectoryReader(
            input_files=[input_path], 
            file_extractor=file_extractor
        ).load_data()
        
        all_docs = "\n\n".join(doc.text for doc in documents)
        
        with open(output_path, "w") as f:
            f.write(all_docs)
        
        logger.info(f"Successfully processed: {os.path.basename(input_path)}")
        return True
    except Exception as e:
        logger.error(f"Error processing {os.path.basename(input_path)}: {str(e)}")
        return False

def main():
    raw_file_dir = os.getcwd()
    input_raw_file_dir = os.path.join(raw_file_dir, "input_docs")
    output_raw_file_dir = os.path.join(raw_file_dir, "parsed_docs")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_raw_file_dir, exist_ok=True)
    
    input_files = get_files_in_directory(input_raw_file_dir)
    logger.info(f"Found {len(input_files)} files to process")
    
    # Prepare processing information for each file
    processing_info = []
    for input_file in input_files:
        input_path = os.path.join(input_raw_file_dir, input_file)
        output_path = os.path.join(
            output_raw_file_dir, 
            f"{os.path.splitext(input_file)[0]}_parsed.txt"
        )
        file_extension = os.path.splitext(input_file)[1]
        processing_info.append((input_path, output_path, file_extension))
    
    # Process files in parallel batches
    successful = 0
    with ThreadPoolExecutor(max_workers=BATCH_SIZE) as executor:
        results = list(executor.map(process_document, processing_info))
        successful = sum(results)
    
    logger.info(f"Processing complete. Successfully processed {successful}/{len(input_files)} files")

if __name__ == "__main__":
    main()