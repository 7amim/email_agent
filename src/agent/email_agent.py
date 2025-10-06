import pandas as pd
import logging

from tqdm.asyncio import tqdm_asyncio
from gpt4all import GPT4All

from src.constants import DEFAULT_LLM
from src.prompts.prompts import EmailAgentPrompt
from src.helpers import extract_fields_from_raw_result

logger = logging.getLogger(__name__)

class EmailAgent:
    def __init__(self, model: str = DEFAULT_LLM):

        self.model = model
        self.llm = GPT4All(model)

        logger.info(f"Initialized model {self.model}")

    async def invoke(self, df: pd.DataFrame):
        logger.info(f"Processing {len(df)} emails using agent...")
        tasks = [self.process_row(row) for row in df.itertuples()]
        results = await tqdm_asyncio.gather(*tasks)

        return results
    
    async def process_row(self, row: pd.Series):
        with self.llm.chat_session():

            subject = row.Subject
            sender = row.Sender
            prompt = EmailAgentPrompt.build(subject, sender)
            response = self.llm.generate(prompt)

        return response

    async def run(self, df: pd.DataFrame) -> pd.DataFrame:

        results = await self.invoke(df)
        extracted_results = extract_fields_from_raw_result(results)
        results_df = pd.DataFrame(extracted_results)
        
        df = pd.concat([df, results_df], axis=1)

        return df
