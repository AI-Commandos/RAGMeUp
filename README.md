# GraphRAG: Extending RAG Me Up with Neo4j Graph Integration

This project introduces **GraphRAG**, an extension to the [RAG Me Up framework](https://github.com/AI-Commandos/RAGMeUp), by integrating Neo4j as a graph database into the RAG pipeline. GraphRAG leverages graph-based data storage and querying capabilities, enabling improved data relationships and enhanced retrieval.

GraphRAG is created for an assignment of the Natural Language Processing course of JADS by **Tjielke Nabuurs, Jeroen Dortmans and Luuk Jacobs of group 17.** Our main goal of the assignment was to modify one of the components of the RAG pipeline that is used by default by the RAG Me Up repository.

# Motivation

Our academic interest lies in graph-based retrieval systems given how graph structures can represent complex relationships and interconnections in a way that mirrors human logic and knowledge. Graph databases like Neo4j are not only highly intuitive for modeling complex systems but also incredibly powerful for performing traceable, structured queries. By using graphs, we can represent the inherent relationships in data and make use of their attributes, enriching the data upon retrieval.

Especially in production environments, reliability and interpretability are paramount. GraphRAG's graph-based approach offers:

1. **Enrichment of Retrieved Data**: GraphRAG not only retrieves the target information but also leverages the graph structure to enrich it with additional relevant attributes.
2. **Traceability**: Each retrieval result is directly linked to its origin in the graph, making it easy to verify and understand why specific information was selected.
3. **Logical Consistency**: The graph structure enforces relationships and dependencies, ensuring that results adhere to the logical connections defined within the data. Which ensure more reliable responses and reduce hallucinations.

# Overview of Changes

We modified the RAG Me Up framework to include a Neo4j-based graph integration as part of the retrieval pipeline. Specifically, we:

1. **Integrated Neo4j** as a new component for data retrieval, allowing the framework to utilize graph-based queries.
2. **Added a new type of dataset and use-case**, demonstrating how a graph structure could enhance retrival insights from the data.
3. **Implemented a loader** to import a csv file into a Neo4j graph database, creating a graph structure with nodes and relationships based on the dataset.

# Dataset Overview

The use-case of GraphRAG is developed on a dataset that consists of customer feedback data, collected through an NPS (Net Promoter Score) survey. It includes the following elements:

1. **Customer-Type Attribute**:  
   A classification of customers based on predefined categories, such as demographic groups or whether they are business or individual customers. This attribute helps segment responses to analyze patterns specific to customer types.

2. **NPS Score (Net Promoter Score)**:  
   A numerical score (0-10) obtained through the question:  
   _"On a scale of 0 to 10, how likely are you to recommend [Organization/Brand] to a friend or colleague?"_

   - **Promoters**: Scores of 9–10 indicate highly satisfied customers.
   - **Passives**: Scores of 7–8 represent neutral customers.
   - **Detractors**: Scores of 0–6 reflect dissatisfied customers.

3. **Quote of the Feedback**:  
   Customers’ responses to the follow-up question:  
   _"Please tell us more about why you wouldn’t recommend [Organization]."_  
   These responses provide qualitative insights into customer satisfaction and dissatisfaction.

4. **Topics Detected in the Quote**:  
   Automatically identified themes or topics in the feedback, such as 'Relieving of concerns', or 'Relationship with your point of contact'. Topics are derived using natural language processing (NLP) and help cluster feedback for actionable analysis.

# Applications:

- **Customer Segmentation**: Analyze NPS scores and feedback by customer type to identify unique trends.
- **Root Cause Analysis**: Use detected topics to pinpoint recurring issues.
- **Actionable Insights**: Combine numerical scores and textual feedback to prioritize service or product improvements.
