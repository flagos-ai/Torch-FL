# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torch.utils.data import Dataset


class DummyTextDataset(Dataset):
    """Simple dummy text dataset for testing training workflow"""

    def __init__(self, tokenizer, num_samples=100, max_length=256):
        self.tokenizer = tokenizer
        self.num_samples = num_samples
        self.max_length = max_length

        self.texts = [
            "Large language models are neural networks trained on vast amounts of text data.",
            "Machine learning is a subset of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses multiple layers of neural networks.",
            "Transformers revolutionized the field of natural language processing.",
            "Attention mechanisms allow models to focus on relevant parts of the input.",
            "Fine-tuning adapts pre-trained models to specific tasks.",
            "Gradient descent is used to optimize neural network parameters.",
            "Backpropagation computes gradients for training neural networks.",
            "The loss function measures how well the model performs.",
        ] * (num_samples // 10 + 1)
        self.texts = self.texts[:num_samples]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        text = self.texts[idx % len(self.texts)]
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(0),
            "attention_mask": encodings["attention_mask"].squeeze(0),
            "labels": encodings["input_ids"].squeeze(0),
        }
