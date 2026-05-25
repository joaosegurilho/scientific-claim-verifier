"""RAG-style response generation using retrieved propositions and chunks."""

from typing import List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from scverifier.config.settings import Config
from scverifier.data.models import Proposition


class ResponseGenerator:
    """Generates AI responses using retrieved context in RAG-style fashion.

    This class handles:
    - Creating context from retrieved propositions
    - Generating AI responses based on context
    - Formatting and extracting source information
    """

    def __init__(self):
        Config.setup_environment()

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE, timeout=Config.LLM_TIMEOUT
        )

        # Create prompt template and chain
        self.prompt = self._create_rag_prompt()
        self.rag_chain = self.prompt | self.llm

    def _create_rag_prompt(self) -> ChatPromptTemplate:
        """Create the RAG prompt template for response generation."""

        system_message = """You are a helpful research assistant. Answer the user's question based on the provided context from scientific documents.

When writing your answer, always base your reasoning on the relationship between the claim and the retrieved evidence, not on external world knowledge.

Guidelines:
- Use only the information provided in the context
- If the context doesn't contain enough information to answer the question, say so
- Be specific and cite relevant details from the context (like this "[Source X]", or "[Sources X, Y, Z]")
- Maintain a scientific and objective tone
- If there are contradictions in the context, acknowledge them

Context:
{context}

Question: {question}

Answer:"""

        return ChatPromptTemplate.from_messages([("system", system_message), ("human", "{question}")])

    # ======================== CORE GENERATION ========================

    def generate_response(self, question: str, retrieved_props: List[Proposition]) -> Dict[str, Any]:
        """Generate an AI response based on the question and retrieved propositions.

        Args:
            question: User's question
            retrieved_props: Proposition domain objects retrieved from vector store

        Returns:
            Dictionary containing question, answer, context, and source information
        """
        # Format context from retrieved propositions
        context = self._format_context(retrieved_props)

        # Generate response using LLM
        response = Config.retry_llm_call(lambda: self.rag_chain.invoke({"context": context, "question": question}))

        return {
            "question": question,
            "answer": response.content,
            "context_used": context,
            "num_sources": len(retrieved_props),
            "sources": self._extract_source_info(retrieved_props),
        }

    # ======================== FORMATTING UTILITIES ========================

    def _format_context(self, propositions: List[Proposition]) -> str:
        """Format retrieved propositions into a context string for the LLM.

        Args:
            propositions: Retrieved Proposition domain objects

        Returns:
            Formatted context string with source citations
        """
        if not propositions:
            return "No relevant context found in the knowledge base."

        context_parts = []
        for i, prop in enumerate(propositions, 1):
            page_info = f"Page {prop.page}" if prop.page else "Page N/A"
            context_parts.append(f"[Source {i} - {prop.source}, {page_info}]: {prop.text}")

        return "\n\n".join(context_parts)

    def _extract_source_info(self, propositions: List[Proposition]) -> List[Dict[str, Any]]:
        """Extract source information from propositions for display.

        Args:
            propositions: Retrieved Proposition domain objects

        Returns:
            List of dictionaries with source metadata
        """
        sources = []

        for prop in propositions:
            # Truncate content for display
            content = prop.text
            if len(content) > 100:
                content = content[:100] + "..."

            source_info = {
                "content": content,
                "page": prop.page,
                "source": prop.source,
                "chunk_id": prop.chunk_id,
                "paper_id": prop.paper_id,
            }
            sources.append(source_info)

        return sources

    def _print_response(self, response_data: Dict[str, Any], retrieval_type: str):
        """Print formatted response with sources.

        Args:
            response_data: Response dictionary from generate_response
            retrieval_type: Type of retrieval used (e.g., "proposition-based")
        """
        print(f"\n AI Response (using {retrieval_type} retrieval):")
        print(f"Question: {response_data['question']}")
        print(f"Answer: {response_data['answer']}")
        print(f"\n Sources used: {response_data['num_sources']}")

        for i, source in enumerate(response_data["sources"], 1):
            page_info = f"Page {source['page']}" if source["page"] else "Page N/A"
            print(f"  {i}. {source['source']} - {page_info}")
            print(f"     Content: {source['content']}")
