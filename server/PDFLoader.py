from typing import Iterator
from langchain_community.document_loaders.pdf import BasePDFLoader
import google.generativeai as genai
from langchain_core.documents import Document
import re

PROMPT = """Please extract all the content from the slides attached, make it very extensive, only repeat information that is in the slides. If there are code snippets summarize them and mention the intention. If there is a visualization please describe it in text form as well. Avoid unnecessary repetition like common logos or names that appear on almost every slide.
Structure your output like this: 
Slide [page number range]: [Content]

Slide [page number range]: [Content]"""


class GeminiPDFLoader(BasePDFLoader):
    def __init__(
        self,
        file_path: str,
    ) -> None:
        super().__init__(file_path, headers=None)
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash")
        self.file = genai.upload_file(self.file_path)
        print("Uploaded file to Google", self.file)

    def lazy_load(self) -> Iterator[Document]:
        response = self.model.generate_content([PROMPT, self.file])
        print(response.text)
        chunks = response.text.split("\n")
        for i, chunk in enumerate(chunks):
            if chunk == "":
                continue

            match = re.search(r"Slides?\s+([\d\s,\+\-]+):\s*(.*)", chunk)
            pageRange = match.group(1).strip() if match else "undefined"
            content = match.group(2).strip() if match else chunk.strip()

            yield Document(
                page_content=content,
                metadata={
                    "source": self.file_path,
                    "page": pageRange,
                    "pk": f"{self.file_path}#{pageRange}",
                },
            )
