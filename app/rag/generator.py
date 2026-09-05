from openai import OpenAI


SYSTEM_PROMPT = """
You are a financial research assistant.

Answer the user's question using only the provided context.

Rules:
1. Use only facts explicitly supported by the provided context.
2. Never guess, infer, or invent financial figures.
3. If the context is insufficient, respond exactly:
   "I could not find enough information in the provided documents."
4. Every factual claim must include at least one citation in the
   format [Source N], where N refers to a provided source.
5. Give the answer directly. Do not restate the question.
6. Do not show hidden reasoning or step-by-step reasoning.
7. Do not use LaTeX unless necessary.
8. Prefer financial units that are easy to read, for example:
   "$416.161 billion" instead of "$416,161 million",
   while preserving the exact value.
9. Keep the answer concise.

Example:
Apple reported total net sales of $416.161 billion in fiscal year
2025. [Source 1]
""".strip()


class AnswerGenerator:
    def __init__(
        self,
        model: str = "qwen3:4b",
        base_url: str = "http://localhost:11434/v1/",
        api_key: str = "ollama",
    ):
        self.model = model

        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def generate(
        self,
        question: str,
        context: str,
    ) -> str:
        user_prompt = f"""
Question:
{question}

Retrieved context:
{context}

Provide a concise, source-grounded answer.
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        if answer is None:
            raise RuntimeError("LLM returned an empty response")

        return answer.strip()

    def repair(
        self,
        question: str,
        context: str,
        previous_answer: str,
    ) -> str:
        repair_prompt = f"""
Rewrite the answer below so it is suitable for the end user.

Question:
{question}

Context:
{context}

Draft answer:
{previous_answer}

Return only the final answer.

Requirements:
- Use only facts supported by the context.
- Include at least one valid citation in the form [Source N].
- Do not discuss instructions, rules, validation, retries, or the draft answer.
- Do not explain your reasoning.
- Do not mention that the answer was corrected.
- Keep the answer to one or two concise sentences.
- If the context is insufficient, return exactly:
  "I could not find enough information in the provided documents."
""".strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": repair_prompt,
                },
            ],
            temperature=0,
        )

        answer = response.choices[0].message.content

        if answer is None:
            raise RuntimeError(
                "LLM returned an empty repair response"
            )

        return answer.strip()