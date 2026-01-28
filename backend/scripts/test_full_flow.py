"""
Test RAG pipeline AND Database Service directly.
"""
import asyncio
from rag_backend.services.rag_pipeline import get_rag_pipeline
from rag_backend.models.chat import ChatQueryRequest
from rag_backend.services.database_service import db_service

async def test_full_flow():
    """Test RAG pipeline and DB storage."""
    print("=" * 60)
    print("Testing Full Flow (RAG + DB)")
    print("=" * 60)
    
    try:
        # 1. Test RAG
        print("1. Initializing RAG pipeline...")
        rag = get_rag_pipeline()
        print("✅ RAG pipeline initialized")
        
        # 2. Test DB Connection
        print("\n2. Testing DB Connection...")
        await db_service.connect()
        print("✅ DB Connected")
        
        # 3. Create Session
        print("\n3. Creating Session...")
        session = await db_service.create_session()
        print(f"✅ Session Created: {session.id}")
        
        # 4. Process Query
        request = ChatQueryRequest(
            query="What is ROS 2?",
            query_type="full_text",
            session_id=session.id
        )
        print(f"\n4. Processing query: {request.query}")
        response = await rag.process_query(request)
        print(f"✅ Answer: {response.answer[:100]}...")
        
        # 5. Save Message
        print("\n5. Saving Message...")
        msg = await db_service.save_message(
            session_id=session.id,
            role='assistant',
            content=response.answer,
            citations=response.citations,
            query_type='full_text'
        )
        print(f"✅ Message Saved: {msg.id}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        await db_service.disconnect()

if __name__ == "__main__":
    asyncio.run(test_full_flow())
