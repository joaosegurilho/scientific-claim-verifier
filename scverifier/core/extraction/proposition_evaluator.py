"""Enhanced proposition quality evaluation using LLMs."""

from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from scverifier.config.settings import Config
from scverifier.data.models import GradePropositions, Proposition, Chunk, PropositionEvaluation


class PropositionEvaluator:
    """Evaluates the quality of generated propositions with enhanced vague statement detection."""

    def __init__(self):
        Config.setup_environment()

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE, timeout=Config.LLM_TIMEOUT)
        self.structured_llm = self.llm.with_structured_output(GradePropositions)

        # Set up prompt
        self.prompt = self._create_evaluation_prompt()
        self.proposition_evaluator = self.prompt | self.structured_llm

    def _create_evaluation_prompt(self) -> ChatPromptTemplate:
        """Create the enhanced proposition evaluation prompt."""

        evaluation_prompt_template = """
            Please evaluate the following proposition based on the criteria below. Pay special attention to detecting vague or meta-commentary statements.

            EVALUATION CRITERIA:
            - **Accuracy**: Rate from 1-10 based on how well the proposition reflects the original text.
            - **Clarity**: Rate from 1-10 based on how easy it is to understand the proposition without additional context.
            - **Completeness**: Rate from 1-10 based on whether the proposition includes necessary details (e.g., dates, qualifiers, specific entities).
            - **Conciseness**: Rate from 1-10 based on whether the proposition is concise without losing important information.

            SPECIAL PENALTIES FOR VAGUE STATEMENTS:
            If the proposition contains any of these characteristics, give it LOW scores (1-3):
            - Meta-commentary about the document itself ("The study found...", "Results showed...", "The paper discusses...")
            - Vague qualifiers without specifics ("significant results", "important findings", "key insights")
            - Process descriptions without concrete facts ("analysis was conducted", "data was collected")
            - Generic conclusions ("implications were discussed", "trends were observed")

            Examples:

            Original Text: "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission."

            Good Proposition: "Neil Armstrong became the first person to walk on the Moon in 1969."
            Evaluation: accuracy: 10, clarity: 10, completeness: 10, conciseness: 10

            Bad Proposition: "The document discusses an important historical event."
            Evaluation: accuracy: 2, clarity: 6, completeness: 1, conciseness: 7

            Original Text: "Cryptotherma noctilucens exhibits bioluminescence through ferrosilicate reduction in oxygen-free environments."

            Good Proposition: "Cryptotherma noctilucens exhibits bioluminescence through ferrosilicate reduction."
            Evaluation: accuracy: 10, clarity: 9, completeness: 9, conciseness: 10

            Bad Proposition: "Key findings were presented about the organism."
            Evaluation: accuracy: 1, clarity: 4, completeness: 1, conciseness: 8

            Original Text: "The researchers conducted extensive analysis of the data and found significant correlations."

            Bad Proposition: "Significant correlations were found."
            Evaluation: accuracy: 6, clarity: 6, completeness: 3, conciseness: 8

            Bad Proposition: "The researchers conducted analysis."
            Evaluation: accuracy: 7, clarity: 6, completeness: 4, conciseness: 9

            Format:
            Proposition: "{proposition}"
            Original Text: "{original_text}"
        """

        return ChatPromptTemplate.from_messages(
            [
                ("system", evaluation_prompt_template),
                ("human", "{proposition}, {original_text}"),
            ]
        )

    def evaluate_proposition(self, proposition: str, original_text: str) -> PropositionEvaluation:
        """Evaluate a single proposition and return evaluation object.

        Args:
            proposition: Proposition text to evaluate
            original_text: Original chunk text for context

        Returns:
            PropositionEvaluation object with scores
        """
        response = Config.retry_llm_call(
            lambda: self.proposition_evaluator.invoke({"proposition": proposition, "original_text": original_text})
        )

        return PropositionEvaluation(
            accuracy=response.accuracy,
            clarity=response.clarity,
            completeness=response.completeness,
            conciseness=response.conciseness,
        )

    def evaluate_propositions(self, propositions: List[Proposition], chunks: List[Chunk]) -> List[Proposition]:
        """Evaluate multiple propositions and return Proposition objects with evaluations.

        Args:
            propositions: List of Proposition objects to evaluate
            chunks: List of Chunk objects for context

        Returns:
            List of Proposition objects with PropositionEvaluation attached
        """
        evaluated_propositions = []

        for proposition in propositions:
            # Get the original chunk content using the chunk_id
            original_text = self._get_chunk_text(chunks, proposition.chunk_id)

            # Evaluate the proposition
            evaluation = self.evaluate_proposition(proposition.text, original_text)

            proposition.evaluation = evaluation
            evaluated_propositions.append(proposition)

        return evaluated_propositions

    def filter_quality_propositions(self, evaluated_propositions: List[Proposition]) -> List[Proposition]:
        """Filter propositions that pass quality checks.

        Args:
            evaluated_propositions: List of Proposition objects with evaluations

        Returns:
            List of Proposition objects that passed quality checks
        """
        return [prop for prop in evaluated_propositions if prop.is_high_quality()]

    def get_failed_propositions(self, evaluated_propositions: List[Proposition]) -> List[Proposition]:
        """Get propositions that failed quality checks.

        Args:
            evaluated_propositions: List of Proposition objects

        Returns:
            List of Proposition objects that failed quality checks
        """
        return [prop for prop in evaluated_propositions if not prop.is_high_quality()]

    def _get_chunk_text(self, chunks: List[Chunk], chunk_id: str) -> str:
        """Get the text from chunks by chunk ID.

        Args:
            chunks: List of Chunk objects
            chunk_id: Chunk ID to find

        Returns:
            Chunk text or empty string if not found
        """
        for chunk in chunks:
            if chunk.chunk_id == chunk_id:
                return chunk.text

        # Fallback: return empty string if chunk not found
        return ""

    def print_evaluation_summary(self, evaluated_propositions: List[Proposition]):
        """Print a detailed summary of the evaluation results.

        Args:
            evaluated_propositions: List of Proposition objects with evaluations
        """
        total = len(evaluated_propositions)
        passed = sum(1 for prop in evaluated_propositions if prop.is_high_quality())
        failed = total - passed

        print("\n=== Proposition Evaluation Summary ===")
        print(f"Total propositions: {total}")
        print(f"Passed quality check: {passed}")
        print(f"Failed quality check: {failed}")
        print(f"Success rate: {passed/total*100:.1f}%" if total > 0 else "Success rate: 0.0%")

        if failed > 0:
            print("\n=== Failed Propositions ===")
            failed_props = self.get_failed_propositions(evaluated_propositions)
            for i, prop in enumerate(failed_props):
                print(f"{i+1}) Proposition: {prop.text}")
                if prop.evaluation:
                    print(f"   Scores: {prop.evaluation.to_dict()}")
                    print(f"   Average: {prop.evaluation.average_score():.1f}/10")
                print()
