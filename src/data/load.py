import re
import json
from datasets import load_dataset, Dataset

class HFDatasetLoader:
    def __init__(self, dataset_name: str, subset_name: str = None, system_prompt: str = ''):
        # dataset name
        self.dataset_name = dataset_name
        
        # sub-set name
        self.subset_name = subset_name
        
        # instruction prompt
        self.system_prompt = system_prompt

    def create_conversation_text(self, data_point):
        text = ""
        for item in data_point["log"]:
            user = self.clean_text(item["user utterance"])
            text += f"user: {user.strip()}\n"

            agent = self.clean_text(item["system response"])
            text += f"agent: {agent.strip()}\n"

        return text

    def clean_text(self, text):
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@[^\s]+", "", text)
        text = re.sub(r"\s+", " ", text)
        return re.sub(r"\^[^ ]+", "", text)

    def generate_training_prompt(self, conversation, summary):
        return f"""### Instruction: {self.system_prompt}\n
                ### Input:
                {conversation.strip()}\n

                ### Response:
                {summary}
                """.strip()
    
    def generate_inference_prompt(self, conversation):
        return f"""### Instruction: {self.system_prompt}\n
                ### Input:
                {conversation.strip()}\n

                ### Response:
                
                """.strip()

    def generate_text(self, data_point):
        summaries = json.loads(data_point["original dialog info"])["summaries"]["abstractive_summaries"]
        summary = " ".join(summaries[0])
        conversation_text = self.create_conversation_text(data_point)
        return {
            "conversation": conversation_text,
            "summary": summary,
            "text": self.generate_training_prompt(conversation_text, summary),
        }

    def process_dataset(self, data):
        return (
            data.shuffle(seed=42)
            .map(self.generate_text)
            .remove_columns(
                [
                    "original dialog id",
                    "new dialog id",
                    "dialog index",
                    "original dialog info",
                    "log",
                    "prompt",
                ]
            )
        )

    def process(self):
        # load dataset from huggingface
        dataset = load_dataset(self.dataset_name, self.subset_name)

        # get training set
        dataset["train"] = self.process_dataset(dataset["train"])

        # get validation set
        dataset["validation"] = self.process_dataset(dataset["validation"])
        
        return dataset

class QADatasetLoader:
    def __init__(self, dataset_name: str, subset_name: str = None, system_prompt: str = '', test_size: float = 0.2, seed: int = 42):
        # dataset name
        self.dataset_name = dataset_name
        
        # sub-set name
        self.subset_name = subset_name
        
        # instruction prompt
        self.system_prompt = system_prompt

        # split size
        self.test_size = test_size

        # split seed
        self.seed = seed

    def generate_training_prompt(self, context, question, answer):
        return f"""### Instruction: {self.system_prompt}\n
                ### Context:
                {context.strip()}\n

                ### Question:
                {question.strip()}

                ### Answer:
                {answer}
                """.strip()
    
    def generate_inference_prompt(self, context, question):
        return f"""### Instruction: {self.system_prompt}\n
                ### Context:
                {context.strip()}\n

                ### Question:
                {question.strip()}

                ### Answer:
                
                """.strip()

    def generate_text(self, data_point):
        # context
        context = data_point.get('context')
        
        # question
        question = data_point.get('question')
        
        # answer
        answer = data_point.get('answers').get('text')[0]

        return {
            "context": context,
            "question": question,
            "answer": answer,
            "text": self.generate_training_prompt(context, question, answer),
        }

    def process_dataset(self, data):
        return (
            data.shuffle(seed=self.seed)
            .map(self.generate_text)
        )

    def process(self):
        # load dataset from huggingface
        if self.subset_name is not None:
            dataset = load_dataset(self.dataset_name, self.subset_name, split='train')
        else:
            dataset = load_dataset(self.dataset_name, split='train')

        # split
        dataset = dataset.train_test_split(test_size=self.test_size)

        # get training set
        dataset["train"] = self.process_dataset(dataset["train"].select(range(5_000)))

        # get validation set
        dataset["validation"] = self.process_dataset(dataset["test"].select(range(500)))
        
        return dataset