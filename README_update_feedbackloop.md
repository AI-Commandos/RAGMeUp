# RAG Me Up
This Readme explains the updates made to the RAG Me Up pipeline. Details on the original pipeline can be found in the `README.md` file. The updates focus on integrating an active user feedback loop into the pipeline. 

# How does the active user feedback loop work?
The goal of having the user provide feedback is to improve the pipeline. This can be done in two ways:
* Update the document retrieval and reranker with a feedback score integrated in the documents.
* The Chat LLM can be updated with the feedback. 

Both ways influence the final answer, and can help improve it. 

The updated RAG pipeline is visualized in the image below:
![RAG pipeline drawing](./Updated_ragmeup.drawio.svg)

The following changes were made compared to the original RAG pipeline:
1. A user can provide feedback to an answer.
2. The feedback is stored in a feedback database.
3. The feedback is added to the documents as they are retrieved in step 2.
4. The reranker takes the feedback into account by adding the weighted feedback scores to the relevance score.
5. The Chat LLM is updated based on the feedback.


# Google Colab Instructions
This section provides detailed instructions for running the code in Google Colab. These steps are essential for initializing the environment, handling the feedback database, and executing the fine-tuning pipeline and server.



### **1. Setting Up the Feedback Database**

The feedback database (`feedback.db`) is necessary for storing user feedback. The following steps are used to set it up in Google Colab:

1. Import the `sqlite3` library to interact with SQLite databases.
2. Connect to the `feedback.db` file:
   - If the file does not exist, it will be created automatically.
3. Create the `Feedback` table with the following structure:
   - **Columns**:
     - `query` (TEXT): The user query.
     - `answer` (TEXT): The assistant's response.
     - `document_id` (TEXT): Identifier for the associated document.
     - `rating` (INTEGER): The feedback rating.
     - `timestamp` (DATETIME): Automatically set to the current time.

The database will persist across sessions when properly saved back to Google Drive.


### **2. Initializing Google Colab Environment**

To integrate files and manage data, follow these steps:

1. **Mount Google Drive**:
   - Use the `drive.mount()` function to access files stored in Google Drive.

2. **Copy Required Files**:
   - Transfer documents (`.pdf` files) to the appropriate directory in the server:
     - Example: `/content/RAGMeUp/server/data/`.
   - Copy the `feedback.db` file from your Google Drive to the server directory.

3. **Verify Files**:
   - List the contents of the `data` directory to ensure that the files are available.



### **3. Running the Fine-Tuning Pipeline**

To execute the fine-tuning pipeline:
- Run the following command:
`!python start_services.py fine-tuning`
- This triggers the fine-tuning process based on user feedback stored in the database.



### **4. Running the Server**

To start the server and interact with the Chat LLM:
- Run the following command:`!python start_services.py server`

- The server will dynamically load the latest fine-tuned model (if available) or the default base model.



### **5. Saving Feedback**

To persist the updated `feedback.db` file after completing your session:
- Copy the file back to your Google Drive: `!cp /content/RAGMeUp/server/feedback.db /content/drive/MyDrive/NLP/feedback/`
 This ensures that feedback data is not lost and can be used for future fine-tuning sessions.



# Details on implementation
For each step mentioned above we will describe how we implemented it.

## User feedback
The original RAG Me Up pipeline already included some basics, such as a thumbs up and thumbs down button in the frontend, as well as some functions in the backend. However, the buttons were not functioning yet, as the functions did not do anything yet. Therefore, this needed to be added. We implemented the `feedfack` function in the controller (`HomeController.scala`), which is called upon as soon as a user clicks one of the feedback buttons. This function takes the feedback data and sends it along so it can be saved. Additionally, we ensured that both feedback buttons were clickable, and also made it easier for users to know this by adding a pointer cursor. We wanted users to only be able to provide feedback once per answer. Once either of the feedback buttons is clicked they both become unclickable, the color opacity of both buttons goes down, and a 'Thank you for your feedback!' message appears underneath the buttons. To ensure the user can still see what button they pressed, the other button gets an even lower opacity.

## Feedback database
To be able to use the feedback for improvements we not only needed to know the answer and the feedback, but also the query and what documents were used in the answer. We also wanted to save the timestamp, as this allows for the tracking of feedback over time, and might show improvements in the feedback. Thus, a feedback database was created using `sqlite`, with the following columns:
* Query (text)
* Answer (text)
* Document_id (text > filename)
* Rating (integer)
* Timestamp (datetime)

The feedback data is collected once a feedback button is clicked, and sent to the controller. The controller then saves it to the server, which calls the `save_feedback` function in `server.py`. This function connects to the database and saves the feedback data.

The database was initially created in the Google Colab notebook, and then saved to a Google Drive folder. Saving it in Google Drive ensures the feedback is saved even when the server is shut down. Before starting the server, the copy of the feedback database stored in Google Drive is loaded in the notebook. After the server is closed, the updated feedback database can be copied to Google Drive again.


## Reranker

The Reranker improves document relevance by integrating user feedback into the ranking process. This README outlines how user feedback is incorporated and used to rerank documents.



### 1. Feedback and Document Retrieval

The Reranker retrieves user feedback from the `feedback.db` database. This feedback includes fields like `query`, `answer`, `document_id`, and `rating`, which represent user evaluations of document relevance.

In addition, the Reranker retrieves all available documents from the `data` directory. The documents are filtered based on supported file types (`txt`, `csv`, `pdf`, `json`) to ensure only relevant files are considered.

### 2. Combining Documents and Feedback

A unified dataframe is created to merge all unique documents with their feedback scores. The logic works as follows:
- Each document in the dataset is checked for corresponding feedback ratings.
- If feedback is available, the ratings are summed up for that document. 
- If no feedback is present, the rating defaults to 0.

This results in a single dataframe where each document is associated with its cumulative feedback score, or 0 if no feedback exists.



### 3. Calculating Relevance Scores

To rank documents effectively, the Reranker calculates a Relevance Score for each document by combining the BM25 Weight and the User Feedback Weight. This is done using the following formula:

*Relevance Score = α ⋅ BM25 Weight + β ⋅ Feedback Weight*

Parameters:
- **BM25 Weight**: The relevance score for a document based on the BM25 algorithm, which uses term frequency and inverse document frequency to measure query-document similarity.
- **Feedback Weight**: The cumulative user feedback rating for a document.
- **α (alpha)**: The weight given to BM25 scores (set to 0.7).
- **β (beta)**: The weight given to user feedback scores (set to 0.3).

These weights prioritize BM25 because it provides a robust measure of text relevance to the query, while feedback adds user-specific insights.

### 4. Sorting and Output
After computing the relevance scores for all documents, the dataframe is sorted in descending order of relevance score. The top-ranked documents are considered the most relevant.

## Remarks on Reranker

Unfortunately, we were not able to successfully implement the reranker with user feedback in the RAGMeUp framework. Despite trying many approaches, we ran into several challenges that stopped us from finishing the task.

### Challenges Faced

### 1. Difficulty Integrating the Reranker
- We created a separate class for the reranker (`Reranker`) to handle feedback-based reranking. However, trying to fit it into the RAGMeUp framework was difficult.
- The reranker needed important inputs like the query, the documents, and the feedback database. Managing these inputs within the framework was a big challenge.
- Getting the feedback from the database and using it in the reranker’s scoring system was also difficult:
  - Feedback and BM25 Score: We were able to get feedback and the BM25 score working in the custom class (`Reranker`), but doing this inside the framework itself wasn’t possible. The framework didn’t make it easy to access the required data (feedback, BM25 score) at the right time.
  - The logic to retrieve feedback was in place, but making sure it was in the correct format and available when needed in the reranker was complex.
  - Combining feedback scores with BM25 or cross-encoder scores took a lot of effort but was hard to check and adjust because of the other integration problems.

### 2. Query Retrieval
- The reranker needs a query to calculate relevance scores. We managed to get the query at some point, but the output with the reranked documents wasn’t working as expected.

### 3. Adjustments to `ScoredCrossEncoderReranker.py`
- We made changes to `ScoredCrossEncoderReranker.py` to add feedback retrieval, document retrieval, and score calculation. But:
  - Managing dependencies like the feedback database, documents, and the query was still a struggle.
  - Testing and validating these changes was difficult because there were issues with how the components were integrated, leading to mismatches between expected and actual inputs.
  - Some attempts can be seen in `ScoredCrossEncoderReranker_original.py` and `scored_cross_encoder_reranker.py`

### 4. Testing and Debugging
- Testing the reranker, whether on its own or within the RAGMeUp framework, was challenging. The lack of clear points to test or easy ways to inject dependencies made it hard to verify each part.
- Errors related to misconfigured dependencies, abstract class constraints, and input mismatches made debugging more difficult and slowed down the process.

## Conclusion

The challenges above highlight the difficulty of integrating feedback-based reranking into the RAGMeUp framework. Key issues included managing dependencies, dealing with abstract class constraints, handling Pydantic errors (`__fields_set__` and `arbitrary_types_allowed`), and testing problems. While we made progress in structuring the reranker and adding feedback logic, we couldn’t get a working implementation due to these issues. Feedback retrieval and BM25 scoring were possible within the custom `Reranker` class, but fitting these steps into the larger RAGMeUp framework was too difficult because of the framework’s limitations and integration issues.


## Chat LLM
To enhance the relevance and performance of the Chat LLM, a fine-tuning mechanism was implemented that integrates user feedback. This document explains the workflow and implementation details.

### 1. Feedback Analysis

The fine-tuning process begins with analyzing user feedback stored in the `feedback.db` database. This is handled by the `analyze_feedback` method in the `LLMFinetuner` class (`fine_tuning_system.py`).

- Feedback entries are filtered based on a configurable timeframe (e.g., 7 or 30 days).
- Each feedback entry includes:
  - **Query**: The user’s input.
  - **Answer**: The assistant’s response.
  - **Document ID**: The associated document.
  - **Rating**: The feedback score.
- The system aggregates feedback to calculate:
  - `feedback_count`: Number of feedback instances.
  - `avg_rating`: Average feedback rating.
- Only high-quality feedback (e.g., `avg_rating > 0.5`) is selected for fine-tuning, ensuring the model learns from reliable data.

### 2. Preparing Training Data

Filtered feedback is transformed into a Hugging Face-compatible dataset using the `prepare_training_data` method. Each training example consists of:
- **User Query**.
- **Document Context**: Extracted from the associated `document_id`.
- **Assistant Response**.

These elements are combined into formatted prompts. Tokenization is applied to structure the data into input tensors, using truncation and padding settings to ensure compatibility with the LLM.

### 3. Model Fine-Tuning

The fine-tuning process is executed by the `fine_tune` method, leveraging the Hugging Face Trainer API. Key steps include:
- **Loading the Base Model**: The system uses `Meta-Llama-3.1-8B-Instruct` as the base model, applying quantization for efficient training on limited resources.
- **Parameter-Efficient Fine-Tuning (PEFT)**:
  - Low-Rank Adaptation (LoRA) is applied to optimize specific layers (e.g., `q_proj`, `v_proj`).
  - This approach reduces the computational overhead.
- **Training Process**:
  - A dataset prepared from feedback is used for training over a defined number of epochs and batch size.
  - Training configurations include:
    - Learning rate: `1e-4`.
    - Epochs: Configurable (e.g., 2).
    - Batch size: Configurable (e.g., 1).
- **Saving the Fine-Tuned Model**:
  - The updated model and tokenizer are saved to the `fine_tuned_model` directory for deployment.

### 4. Periodic Execution

The fine-tuning process is automated using the `fine_tuning_scheduler.py` script, which is orchestrated by the `start_services.py` file. This setup ensures:
- **Scheduled Updates**: (This is commented out as we wanted to test system immnediately)
  - The pipeline runs at regular intervals (e.g., weekly).
  - Scheduling can be configured using a time-based scheduler.
- **Immediate Updates**:
  - Fine-tuning is also triggered on server startup.
- **Error Handling**:
  - Logs and error messages ensure graceful handling of issues.

### 5. Integration and Deployment

The server is configured to automatically detect and load the latest fine-tuned model. This functionality is implemented in the `server.py` file with the following logic:

- **Dynamic RAG Helper Initialization**:
  - Depending on the environment configuration, the server selects either a cloud-based (`RAGHelperCloud`) or local (`RAGHelperLocal`) RAG helper.
  - This selection is determined by environment variables such as `use_openai`, `use_gemini`, `use_azure`, and `use_ollama`.

- **Fine-Tuned Model Detection**:
  - If the `RAGHelperLocal` is used, the server checks for the presence of a fine-tuned model in the specified output directory (default: `./fine_tuned_model`).
  - When a fine-tuned model is found (indicated by the existence of `pytorch_model.bin`), the `RAGHelperLocal` is instantiated with the path to the fine-tuned model.

- **Fallback Logic**:
  - If no fine-tuned model is found, the `RAGHelperLocal` uses the default base model.


## Remarks on the Chat LLM Fine Tuner

Unfortunately at the last week we figured that the finetuned model was too big to be runned on the GPU of Google Colab. We made it work using quantization, Parameter-Efficient Fine-Tuning (PEFT) and Memory Optimization:

Quantization:
- Used BitsAndBytesConfig to load the model in 4-bit quantization

Parameter-Efficient Fine-Tuning (PEFT):
- Implemented LoRA (Low-Rank Adaptation)
- Only fine-tunes a small number of parameters

Memory Optimization:
- Reduced sequence length to 128
- Set batch size to 1 (as suggested by your professor)

Reduced Training Parameters:
- Lowered epochs to 1
- Reduced save frequency
- Minimized save total limit

With these modification we made it able to make the finetuning work. However due to the time limit we cound't solve the integration. The model is created and filed in the right directory. However during the last test we noticed it didn't find the `pytorch_model.bin`. The time runned out towards the deadline so we couldn't solve this last problem before deadline.

