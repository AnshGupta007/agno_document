#!/usr/bin/env python3
"""
Agno Document Q&A Agent - Using Groq model with local embeddings
This agent uses Groq for fast LLM operations and local embeddings for document Q&A
"""

import json
from datetime import datetime
from agno.agent import Agent
from agno.models.groq import Groq
from local_document_search import search_local_documents
from dotenv import load_dotenv

load_dotenv()

def get_current_datetime() -> str:
    """Get the current date and time for context"""
    current = datetime.now()
    return json.dumps({
        "current_date": current.strftime("%Y-%m-%d"),
        "current_time": current.strftime("%H:%M"),
        "current_datetime": current.strftime("%Y-%m-%d %H:%M:%S"),
        "day_of_week": current.strftime("%A")
    })


def search_documents_tool(query: str) -> str:
    """
    Search through uploaded documents using local models to find relevant information
    
    Args:
        query: User's question about the documents
        
    Returns:
        JSON string with relevant context and sources
    """
    return search_local_documents(query)


def create_agno_document_agent(model_name: str = "openai/gpt-oss-20b"):
    """
    Create an Agno agent for document Q&A using Groq model
    
    Args:
        model_name: Name of the Groq model to use
        
    Returns:
        Configured Agno Agent instance
    """
    
    # Create the agent with Groq model
    agent = Agent(
        model=Groq(id=model_name),
        tools=[search_documents_tool],
        instructions="""You are a helpful and knowledgeable document Q&A assistant. Your primary role is to answer questions based on the content of uploaded documents using RAG (Retrieval-Augmented Generation).

Your personality:
- Professional yet friendly and approachable
- Accurate and precise in your responses
- Multilingual - you can answer questions in any language the user asks
- Always cite your sources when providing information
- Honest about limitations - if you don't know something or if the documents don't contain the information, you say so clearly

Your capabilities:
1. Answer questions based on document content using semantic search
2. Provide detailed explanations with proper source attribution
3. Handle queries in multiple languages
4. Synthesize information from multiple document chunks
5. Clarify when information is not available in the documents

How to respond:
- Always use the search_documents_tool function for any user question
- When you receive context from the function, use it to provide a comprehensive answer
- Always mention which documents/sources your answer comes from
- If the context is insufficient, clearly state what information is missing
- Be conversational but maintain accuracy
- For multilingual queries, respond in the same language as the question

Response format:
- Provide a clear, direct answer to the question
- Include relevant details from the context
- List the source documents at the end
- If answering in a non-English language, maintain the same structure

Example responses:
- "Based on the documents, [answer]. This information comes from [source files]."
- "According to the documentation in [filename], [detailed answer]."
- "I found information about this in [sources], which indicates that [answer]."

IMPORTANT: 
- Always call search_documents_tool for ANY user question
- Use the context provided to give accurate, source-backed answers
- Don't make up information not present in the documents
- Be clear about the confidence level of your answers"""
    )
    
    return agent


def main():
    """Main function to run the Agno document Q&A agent"""
    print("📚 Document Q&A Assistant (Agno + Groq)")
    print("=" * 60)
    print("Powered by fast Groq models with local embeddings!")
    print("=" * 60)
    print("Features:")
    print("• Fast Groq models for language understanding")
    print("• Local SentenceTransformer embeddings")
    print("• High-speed document search and Q&A")
    print("• Multi-language support")
    print("• Local embeddings for privacy")
    print("\nTo get started, make sure you have:")
    print("1. GROQ_API_KEY environment variable set")
    print("2. Documents in the 'agno_document_qa/documents' folder")
    print("3. Supported formats: .txt, .md, .json, .docx, .py, .js, .html, .css")
    print("\nType 'quit', 'exit', or 'bye' to end the conversation")
    print("=" * 60)

    # Create the agent
    try:
        agent = create_agno_document_agent()
        print("✅ Groq AI agent initialized successfully!")
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("Make sure GROQ_API_KEY environment variable is set")
        return

    # Interactive loop
    while True:
        try:
            user_input = input("\n🧑 You: ")
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\n👋 Thank you for using the Local Document Q&A Assistant!")
                break
            
            if not user_input.strip():
                continue
            
            print("\n🤖 Assistant: ")
            # Use Agno's print_response method
            agent.print_response(user_input)
            
        except KeyboardInterrupt:
            print("\n\n👋 Thanks for using the Local Document Q&A Assistant!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or rephrase your request.")


if __name__ == "__main__":
    main()
