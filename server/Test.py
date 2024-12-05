from rerank_documents_with_feedback import Reranker

def periodic_fine_tuning():
    """
    Run fine-tuning process
    """
    try:
        fine_tuner = Reranker()
        fine_tuner.rerank_documents_with_feedback(
            query, documents
        )
    except Exception as e:
        print(f"Fine-tuning failed: {e}")

def main():
    # Run fine-tuning immediately on startup
    periodic_fine_tuning()

if _name_ == "_main_":
    main()