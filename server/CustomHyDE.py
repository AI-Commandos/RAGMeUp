from langchain.chains import HypotheticalDocumentEmbedder, LLMChain

class CustomHypotheticalDocumentEmbedder(HypotheticalDocumentEmbedder):
    def embed_query(self, text: str, return_text: bool = False) -> str | list[float]:
        """
        Generate a hypothetical document and return either the raw document text
        or the final embeddings.
        """
        # Extract the input key
        var_name = self.input_keys[0]

        # Generate the hypothetical document using the LLM
        result = self.llm_chain.invoke({var_name: text})
        if isinstance(self.llm_chain, LLMChain):
            documents = [result[self.output_keys[0]]]
        else:
            documents = [result]

        # Return the raw hypothetical document text if requested
        if return_text:
            return documents[0]

        # Embed the hypothetical document
        embeddings = self.embed_documents(documents)

        # Combine embeddings and return
        return self.combine_embeddings(embeddings)
