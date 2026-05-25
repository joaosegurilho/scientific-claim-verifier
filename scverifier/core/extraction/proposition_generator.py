"""Improved proposition generation from document chunks using LLMs."""

from typing import List
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from scverifier.config.settings import Config
from scverifier.data.models import GeneratePropositions, Proposition, Chunk
from scverifier.utils.id_generator import get_next_prop_id


class PropositionGenerator:
    """Generates propositions from document chunks using LLMs.

    Works directly with Chunk domain objects - no LangChain Documents needed.
    """

    def __init__(self):
        Config.setup_environment()

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(model=Config.LLM_MODEL, temperature=Config.LLM_TEMPERATURE, timeout=Config.LLM_TIMEOUT)
        self.structured_llm = self.llm.with_structured_output(GeneratePropositions)

        # Set up prompt
        self.prompt = self._create_prompt()
        self.proposition_generator = self.prompt | self.structured_llm

    def _create_prompt(self) -> ChatPromptTemplate:
        """Create the improved proposition generation prompt with better few-shot examples."""

        # Improved few-shot examples showing good vs bad propositions
        proposition_examples = [
            {
                "document": "In 1969, Neil Armstrong became the first person to walk on the Moon during the Apollo 11 mission. The mission lasted 8 days and included comprehensive geological sampling. Armstrong collected 21.5 kg of lunar samples for analysis.",
                "propositions": [
                    "Neil Armstrong was an astronaut who participated in the Apollo 11 mission.",
                    "Neil Armstrong became the first person to walk on the Moon in 1969.",
                    "The Apollo 11 mission occurred in 1969.",
                    "The Apollo 11 mission lasted 8 days.",
                    "Armstrong collected 21.5 kg of lunar samples during the Apollo 11 mission.",
                    "The Apollo 11 mission included comprehensive geological sampling of the Moon.",
                ],
            },
            {
                "document": (
                    "In 1247 of the Third Age, Queen Elara forged the Obsidian Pact with dragon Valthorax during the Night of Whispers, binding Eldermere to a 300-year alliance. "
                    "The pact required Eldermere to supply 500 barrels of enchanted starmetal annually. Researchers noted significant results in three areas but found the methodology well-designed."
                ),
                "propositions": [
                    "Queen Elara forged the Obsidian Pact with dragon Valthorax in 1247 of the Third Age.",
                    "The Obsidian Pact was signed during the Night of Whispers.",
                    "The Obsidian Pact bound the kingdom of Eldermere to a 300-year alliance.",
                    "The Obsidian Pact required Eldermere to supply 500 barrels of enchanted starmetal annually to Valthorax."
                ],
            },
            {
                "document": "In this study, we found significant reduction in mortality (p<0.05) and improved quality of life across 500 patients over 2 years.",
                "propositions": [
                    "Mortality was significantly reduced (p<0.05)",
                    "Quality of life improved across patients",
                    "Study included 500 patients",
                    "Study duration was 2 years",
                ],
            },
            {
                "document": "The study found significant results in three key areas. Researchers noted important implications for future work. The methodology was well-designed and appropriate for the research questions.",
                "propositions": [],  # Empty because all statements are vague
            },
        ]

        # Convert proposition lists to strings for the template
        for example in proposition_examples:
            example["propositions_str"] = str(example["propositions"])

        example_proposition_prompt = ChatPromptTemplate.from_messages(
            [
                ("human", "{document}"),
                ("ai", "{propositions_str}"),
            ]
        )

        few_shot_prompt = FewShotChatMessagePromptTemplate(
            example_prompt=example_proposition_prompt,
            examples=proposition_examples,
        )

        # system message
        system = """
            Please extract specific, factual propositions from the following text. Each proposition must meet ALL of these criteria:

            REQUIRED CRITERIA:
            1. **Express ONE Specific Fact**: State exactly one verifiable fact or claim, not multiple related facts
            2. **Be Completely Self-Contained**: Understandable without any additional context from the document
            3. **Use Concrete Entities**: Use full names, specific measurements, dates, or concrete entities - avoid pronouns and vague references
            4. **Include Precise Details**: Include relevant dates, quantities, locations, or qualifiers that make the fact specific
            5. **Avoid Meta-Commentary**: Do not describe the document itself (e.g., "The study found...", "Results showed...", "The paper discusses...")
            6. **Be Factually Verifiable**: The statement should be something that could be independently verified or checked

            AVOID THESE TYPES OF VAGUE STATEMENTS:
            - "The research showed significant results"
            - "Important findings were presented"
            - "The methodology was appropriate"
            - "Future research is needed"
            - "The study discussed various aspects"
            - "Key insights were revealed"
            - "The analysis demonstrated trends"

            GOOD EXAMPLES:
            "Neil Armstrong walked on the Moon in 1969 during the Apollo 11 mission"
            "The mitosis cycle of C. noctilucens occurs every 19.7 hours"
            "Ferrivorax umbralis possesses an exoskeleton infused with iron sulfide"

            BAD EXAMPLES:
            ✗ "The organism was studied extensively"
            ✗ "Important characteristics were observed"
            ✗ "The findings have implications for future research"

            IMPORTANT: You must ALWAYS return your response as a Python list of strings.

            FORMAT REQUIREMENTS:
            - Return ONLY the list, nothing else
            - Do NOT wrap it in code blocks (no ```python or ```)
            - Do NOT add any explanatory text before or after
            - Format it exactly like this: ['proposition 1', 'proposition 2', 'proposition 3']
            - If there are no valid propositions, return an empty list: []
        """

        return ChatPromptTemplate.from_messages(
            [
                ("system", system),
                few_shot_prompt,
                ("human", "{document}"),
            ]
        )

    def generate_propositions_from_chunk(self, chunk: Chunk) -> List[Proposition]:
        """Generate propositions from a single chunk and return Proposition objects.

        Args:
            chunk: Chunk object to extract propositions from

        Returns:
            List of Proposition objects with metadata from the chunk
        """
        try:
            response = Config.retry_llm_call(lambda: self.proposition_generator.invoke({"document": chunk.text}))

            propositions = []
            for proposition_text in response.propositions:
                prop = Proposition(
                    text=proposition_text,
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    paper_id=chunk.paper_id,
                    prop_id=get_next_prop_id(),
                    page=chunk.page,
                    evaluation=None,
                )
                propositions.append(prop)

            return propositions
        except Exception as e:
            print(f"        Failed to extract propositions (timeout/error): {str(e)[:50]}")
            return []

    def generate_propositions_from_chunks(self, chunks: List[Chunk]) -> List[Proposition]:
        """Generate propositions from multiple chunks and return Proposition objects.

        Args:
            chunks: List of Chunk objects to process

        Returns:
            List of all Proposition objects from all chunks
        """
        all_propositions = []

        for i, chunk in enumerate(chunks):
            propositions = self.generate_propositions_from_chunk(chunk)
            all_propositions.extend(propositions)
            print(f"      Analysed chunk {i}. Found {len(propositions)} propositions.")

        return all_propositions
