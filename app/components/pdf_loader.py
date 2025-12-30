import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from app.common.logger import get_logger
from app.common.custom_exception import CustomException

from app.config.config import DATA_PATH, CHUNK_SIZE, CHUNK_OVERLAP

logger = get_logger(__name__)

def load_pdf_files():
    try:
        if not os.path.exists(DATA_PATH):
            raise CustomException("Data path doesnt exist")
        
        logger.info(f"Loading files from {DATA_PATH}")

        documents = []

        # 1. Load PDFs
        try:
            pdf_loader = DirectoryLoader(DATA_PATH, glob="*.pdf", loader_cls=PyPDFLoader)
            pdf_docs = pdf_loader.load()
            if pdf_docs:
                documents.extend(pdf_docs)
                logger.info(f"Loaded {len(pdf_docs)} PDF documents")
        except Exception as e:
            logger.warning(f"Error loading PDFs: {e}")

        # 2. Load Text Files (txt) - Added support for supplemental data
        try:
            txt_loader = DirectoryLoader(DATA_PATH, glob="*.txt", loader_cls=TextLoader)
            txt_docs = txt_loader.load()
            if txt_docs:
                documents.extend(txt_docs)
                logger.info(f"Loaded {len(txt_docs)} TXT documents")
        except Exception as e:
             logger.warning(f"Error loading TXT files: {e}")

        if not documents:
            logger.warning("No documents (PDF or TXT) were found")
        else:
            logger.info(f"Successfully fetched {len(documents)} total documents")

        return documents
    
    except Exception as e:
        error_message = CustomException("Failed to load documents", e)
        logger.error(str(error_message))
        return []
    

def create_text_chunks(documents):
    try:
        if not documents:
            raise CustomException("No documents were found")
        
        logger.info(f"Splitting {len(documents)} documents into chunks")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        text_chunks = text_splitter.split_documents(documents)

        logger.info(f"Generated {len(text_chunks)} text chunks")
        return text_chunks
    
    except Exception as e:
        error_message = CustomException("Failed to generate chunks", e)
        logger.error(str(error_message))
        return []
