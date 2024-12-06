import os
import re
import shutil
from dotenv import load_dotenv
import PyPDF2
from statistics import mean
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
from tqdm import tqdm

load_dotenv()

class SlideDeckSummarizer:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.slide_deck_fps = list()
        self.deck_as_images = {}
        self.output_folder = 'output-rendered-images-of-slides'
        self.allowed_models = {'gemini-1.5-flash', 'gemini-1.5-pro'}

        if os.getenv('use_gemini') == 'True':
            if not os.getenv('gemini_model_name') in self.allowed_models:
                raise ValueError("Use a multimodal model from Gemini: gemini-1.5-flash or gemini-1.5-pro")
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            self.model = genai.GenerativeModel(os.getenv('gemini_model_name'))
        else:
            raise ValueError("Use a multimodal model from Gemini: gemini-1.5-flash or gemini-1.5-pro")
        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def check_slide_deck(file_path) -> bool:
        """Check if a pdf is a slide deck so we know which pdfs to summarize

        Parameters:
            file_path (str): Path to the PDF file.
        """
        try:
            pdf_reader = PyPDF2.PdfReader(file_path)
            num_pages = len(pdf_reader.pages)

            if num_pages == 0:
                raise ValueError("The PDF file is empty.")

            short_text_threshold = 30  # Character limit to consider a text 'short'
            slide_like_pages = 0
            text_document_pages = 0

            for page in pdf_reader.pages:
                text = page.extract_text()

                # Skip if no text was extracted
                if not text:
                    continue

                # Split the text into lines and words
                lines = text.splitlines()
                words = text.split()

                # Metrics to analyze
                avg_words_per_line = mean(len(line.split()) for line in lines if line.strip()) if lines else 0
                avg_chars_per_line = mean(len(line) for line in lines if line.strip()) if lines else 0

                # Determine if the page is more like a slide or a text document
                if avg_words_per_line < 10 or avg_chars_per_line < short_text_threshold:
                    slide_like_pages += 1  # Slide-like characteristics
                else:
                    text_document_pages += 1  # Text document characteristics

            # Percentage of slide-like pages vs. normal text pages
            slide_like_ratio = slide_like_pages / num_pages
            text_document_ratio = text_document_pages / num_pages

            # Classification logic (adjust thresholds as needed)
            if slide_like_ratio > 0.5:
                return True
            return False

        except Exception as e:
            return f"Error analyzing PDF: {str(e)}"

    def _select_slidedecks(self):
        """Selecting the pdfs that are slide decks and storing the filepaths"""
        self.slide_deck_fps = [os.path.join(self.input_folder, file) for file in os.listdir(self.input_folder)
                               if file.endswith('.pdf') and self.check_slide_deck(file)]

    def _convert_slides_to_images(self):
        for path_to_slide_deck in self.slide_deck_fps:

            # Create folder in the output dir for the file
            filename = Path(path_to_slide_deck).stem
            os.makedirs(os.path.join(self.output_folder, filename), exist_ok=True)
            current_output_folder = os.path.join(self.output_folder, filename)

            # converting each slide to a jpeg and saving it in the output folder
            paths = convert_from_path(path_to_slide_deck, output_folder=current_output_folder,
                                      dpi=100, paths_only=True, fmt='jpeg')

            # keeping the paths to the rendered images as an attribute
            self.deck_as_images[path_to_slide_deck] = paths

    def _natural_sort_key(self, key):
        return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', key)]

    def _summarize_all_slides_from_deck(self, deck_filepath) -> str:
        """Method that will create a large string in the format:

        -----SLIDE i-----

        <THE SUMMARY>


        -----SLIDE i+1-----

        <THE SUMMARY>

        """
        image_file_paths = self.deck_as_images[deck_filepath]
        loaded_images = []

        # Sort files naturally and load them
        for file in sorted(image_file_paths, key=self._natural_sort_key):
            loaded_images.append(Image.open(file))

        prompt_1 = ("You are data science student that is following a lecture and making extensive notes of slides for his exam.\
                  General Rules: \
                   - Give the title or topic of the slide. \
                   - Explain the material in way the professor would do it. \
                   - Do not start with \'This slide says\' or anything similar. Rather make a story of it \
                   - If the content of the slide is only one question, answer it.\
                  Sequential Context: \
                   - If a slide introduces entirely new content, summarize it fully. \
                   - If a slide builds on or modifies the content of the previous slide, only describe the new or additional information. \
                  Do not include any additional comments or use the web. \
                  Example:\
                  **Slide 11: WordPiece Tokenization**\
                  It's also important to note that BERT's vocabulary isn't just a simple list of words. It uses WordPiece tokenization, which breaks words into subword units (including whole words, characters, and special subword tokens). This method helps to handle out-of-vocabulary words more effectively. \
                  \n \
                  ********* \
                  \n\
                  **Slide 12 & 13 & 14: Obtaining Word Vectors from BERT**\
                  \n\
                  Getting a word vector from BERT isn't trivial. A word will have different vector representations depending on its context within a sentence.  Various approaches and libraries handle this, but they all rely on some sort of pooling mechanism.  This contextualized nature is what makes BERT so powerful.\
                  Explicitly let me know what you have to do and that you understand it. Also give an example of how you would it.")
        prompt_2 = "Perfect, let's get started! Do not include any additional comments only go through the slides."

        chat_session = self.model.start_chat(
            history=[]
        )

        chat_session.send_message(prompt_1)

        response = chat_session.send_message([prompt_2] + loaded_images)

        return response.text

    def transform_slidedecks_and_remove_pdf(self):

        self._select_slidedecks()
        self._convert_slides_to_images()

        for deck in tqdm(self.deck_as_images.keys()):
            summary = self._summarize_all_slides_from_deck(deck)
            # summary = self._summarize_all_slides_from_deck(deck)
            #
            # write summary to text file
            new_fp = os.path.join(self.input_folder, Path(deck).stem + '.txt')
            with open(new_fp, 'w') as f:
                f.write(summary)

            # remove the pdf from the input dir
            os.remove(deck)
        shutil.rmtree('output-rendered-images-of-slides')

if __name__ == "__main__":
    llm = []
    load_dotenv()
    slide_deck_summarizer = SlideDeckSummarizer('data/slides-test')
    slide_deck_summarizer.transform_slidedecks_and_remove_pdf()

