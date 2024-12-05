import os
from dotenv import load_dotenv
import PyPDF2
from statistics import mean
from pathlib import Path
import base64
from mimetypes import guess_type
from pdf2image import convert_from_path

from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate

load_dotenv('.env')


class SlideDeckSummarizer:

    def __init__(self, input_folder, llm):

        self.input_folder = input_folder
        self.slide_deck_fps = list()
        self.deck_as_images = {}
        self.output_folder = 'output-rendered-images-of-slides'
        self.llm = llm
        self.allowed_models = {'gemini-1.5-flash', 'gemini-1.5-flash-8b', 'gemini-1.5-pro',
                               'gpt-4o', 'gpt-4-turbo'}

        if os.getenv('use_openai') == 'True':
            if not os.getenv('openai_model_name') in self.allowed_models:
                raise ValueError("Use a multimodal model from OpenAI")

        elif os.getenv('use_gemini') == 'True':
            if not os.getenv('gemini_model_name') in self.allowed_models:
                raise ValueError("Use a multimodal model from Gemini")

        os.makedirs(self.output_folder, exist_ok=True)

    @staticmethod
    def check_slide_deck(file_path) -> bool:
        """Check if a pdf is a slide deck so we know which pdfs to summarize

        Parameters:
            file_path (str): Path to the PDF file.

        Returns:
            str: "Normal Text Document" or "Slide Deck"
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

    def _summarize_slide(self, path_to_slide_as_image):

        mime_type, _ = guess_type(path_to_slide_as_image)

        # Default to png
        if mime_type is None:
            mime_type = 'image/png'

        # Read and encode the image file
        with open(path_to_slide_as_image, "rb") as image_file:
            base64_encoded_data = base64.b64encode(image_file.read()).decode('utf-8')

        # Construct the data URL
        encoded_image_url = f"data:{mime_type};base64,{base64_encoded_data}"

        prompt_template = HumanMessagePromptTemplate.from_template(
            template=[
                {"type": "text", "text": "Summarize this image"},
                {
                    "type": "image_url",
                    "image_url": "{encoded_image_url}",
                },
            ]
        )

        summarize_image_prompt = ChatPromptTemplate.from_messages([prompt_template])
        image_chain = summarize_image_prompt | self.llm  # pipe is used in langchain for creating a chain

        return image_chain.invoke(input={"encoded_image_url": encoded_image_url})

    def _summarize_all_slides_from_deck(self, deck_filepath) -> str:
        """Method that will create a large string in the format:

        -----SLIDE i-----

        <THE SUMMARY>


        -----SLIDE i+1-----

        <THE SUMMARY>

        """

        result_summaries = list()
        for slide_image_fp in self.deck_as_images[deck_filepath]:
            llm_summary = self._summarize_slide(slide_image_fp).content
            slide_number = Path(slide_image_fp).stem.split("-")[-1]
            result_summaries.append((slide_number, llm_summary))

        result_summaries.sort(key=lambda x: x[0])

        result_summary_string = str()
        for slide_num, enrichment in result_summaries:
            result_summary_string += f'-----SLIDE {slide_num}-----\n\n'
            result_summary_string += f'{enrichment}\n\n\n'
        return result_summary_string

    def transform_slidedecks_and_remove_pdf(self):

        self._select_slidedecks()
        self._convert_slides_to_images()

        for deck in self.deck_as_images.keys():
            summary = self._summarize_all_slides_from_deck(deck)

            # write summary to text file
            new_fp = os.path.join(self.input_folder, Path(deck).stem + '.txt')
            with open(new_fp, 'w') as f:
                f.write(summary)

            # remove the pdf from the input dir
            os.remove(deck)
