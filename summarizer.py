from transformers import pipeline

def text_summarizer(text, max_length=150):
    # Explicitly use PyTorch implementation
    summarizer = pipeline(
        "summarization", 
        model="facebook/bart-large-cnn",
        framework="pt"  # Force PyTorch usage
    )
    summary = summarizer(
        text,
        max_length=max_length,
        min_length=30,
        do_sample=False,
        truncation=True
    )
    return summary[0]['summary_text']

if __name__ == "__main__":
    input_text = """
    Artificial intelligence (AI) is transforming industries across the globe, 
    enabling machines to perform tasks that once required human intelligence. 
    From autonomous vehicles to virtual assistants, AI technologies are 
    increasingly embedded in daily life. In healthcare, AI helps diagnose 
    diseases and suggest treatments. In finance, it's used for fraud detection 
    and algorithmic trading. Despite its benefits, AI also raises ethical concerns, 
    such as data privacy, job displacement, and algorithmic bias. As development 
    accelerates, governments and organizations are working to ensure responsible 
    and fair deployment of AI technologies.
    """
    
    print("Original Text Length:", len(input_text))
    result = text_summarizer(input_text)
    print("\nSummary:", result)
    print("Summary Length:", len(result))
