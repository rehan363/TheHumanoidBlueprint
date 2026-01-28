"""
Test RAG pipeline directly without API layer.
"""
import asyncio
from rag_backend.services.rag_pipeline import get_rag_pipeline
from rag_backend.models.chat import ChatQueryRequest

async def test_rag_pipeline():
    """Test RAG pipeline directly."""
    print("=" * 60)
    print("Testing RAG Pipeline Directly")
    print("=" * 60)
    
    try:
        # Get RAG pipeline
        print("Initializing RAG pipeline...")
        rag = get_rag_pipeline()
        print("✅ RAG pipeline initialized")
        
        # Create request
        request = ChatQueryRequest(
            query="What is ROS 2?",
            query_type="full_text"
        )
        
        print(f"\nProcessing query: {request.query}")
        
        # Process query
        response = await rag.process_query(request)
        
        print(f"\n✅ Query processed successfully!")
        print(f"Answer: {response.answer[:200]}...")
        print(f"Citations: {len(response.citations)}")
        print(f"Processing time: {response.processing_time_ms}ms")
        
        if response.citations:
            print("\nCitations:")
            for i, citation in enumerate(response.citations[:3], 1):
                print(f"  {i}. {citation.source} (score: {citation.relevance_score:.2f})")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_rag_pipeline())
