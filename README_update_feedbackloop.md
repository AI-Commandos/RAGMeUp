# RAG Me Up
This Readme explains the updates made to the RAG Me Up pipeline. Details on the original pipeline can be found in the `README.md` file. The updates focus on integrating user feedback into the pipeline. 

# How does the user feedback work?
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
4. The reranker takes the feedback into account.
5. The Chat LLM is updated based on the feedback.

# Details on implementation
For each step mentioned above we will describe how we implemented it.

## User feedback
The original RAG Me Up pipeline already included some basics, such as a thumbs up and thumbs down button in the frontend, as well as some functions in the backend. However, the buttons were not functioning yet, as the functions did not do anything yet. Therefore, this needed to be added. We implemented the `feedfack` function in the controller (`HomeController.scala`), which is called upon as soon as a user clicks one of the feedback buttons. This function takes the feedback data and sends it along so it can be saved. Additionally, we ensured that both feedback buttons were clickable, and also made it easier for users to know this by adding a pointer cursor. We wanted users to only be able to provide feedback once per answer. Once either of the feedback buttons is clicked they both become unclickable, the color opacity of both buttons goes down, and a 'Thank you for your feedback!' message appears underneath the buttons. To ensure the user can still see what button they pressed, the other button gets an even lower opacity.

## Feedback database
To be able to use the feedback for improvements we not only needed to know the answer and the feedback, but also the query and what documents were used in the answer. We also wanted to save the timestamp, as this allows for the tracking of feedback over time, and might show improvements in the feedback. Thus, a feedback database was created using `sqlite`, with the following columns:
* Query (text)
* Answer (text)
* Document_id (text)
* Rating (integer)
* Timestamp (datetime)

The feedback data is collected once a feedback button is clicked, and sent to the controller. The controller then saves it to the server, which calls the `save_feedback` function in `server.py`. This function connects to the database and saves the feedback data.

The database was initially created in the Google Colab notebook, and then saved to a Google Drive folder. Saving it in Google Drive ensures the feedback is saved even when the server is shut down. Before starting the server, the copy of the feedback database stored in Google Drive is loaded in the notebook. After the server is closed, the updated feedback database can be copied to Google Drive again.

## Document retrieval

## Reranker

## Chat LLM