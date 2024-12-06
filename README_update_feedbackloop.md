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

To enhance the relevance of the document, the reranker was updated with the feedback retrieved by the user. In this section the process of implementing the user feedback in the reranker is explained. 

### 1. Feedback and Document Retrieval
To be able to implement the user feedback into the reranker, the data is retrieve from the `feedback.db` database and put into a dataframe. To combine the feedback with the documents also the documents need to be retrieved. This is done in a similar way as in the `server.py` from the original framework. 

### 2. Combining documents and their feedback
Next, there is a dataframe created which contains all the unique documents available. If there was feedback available for a document in the dataframe the feedback score got added, if there were multiple feedback values available they got added up. If there was no rating present the feedback score was put as 0. This got but into one dataframe.

### 3. Calculating Relevance Score
Now that there is a dataframe with all the documents and their feedback score. We need to order them to get the most relevant documents. To do this a relevance score was calculated using the following formula: 
Relevance Score = α ⋅ BM25 Weight+ β ⋅ User Feedback Weight 

            bm25_score_weight (float): The BM25 relevance score for a document.
            feedback_weight (float): The user feedback weight for a document.
            alpha (float): Weight of BM25 in the relevance formula.
            beta (float): Weight of user feedback in the relevance formula.

With α = 0.7 and β = 0.3, since we think the BM25 relevance score for a document is more important.  
The BM25 Weight is calculated  for each document based on the query asked using `rank_bm25` . 
User Feedback Weight is determined by the adding up the overal received feedback of all user for that particual document. 

After the relevance score is calculated it is added to the dataframe. Based on the relevance score the dataframe is sorted on the relevance score to get the highest score on top. 



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
