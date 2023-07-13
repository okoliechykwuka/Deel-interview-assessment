# Deel-interview-assessment


Certainly! Here's a brief discussion of the solution and its possible limitations for tasks 1 and 2:

**Task 1: Matching Transactions by Name**

Solution:
- The solution uses fuzzy matching (specifically, token sort ratio) to calculate a match metric between the given name and transaction descriptions.
- The transactions are then sorted in descending order of the match metric to determine relevance.
- The API returns the matched transactions and the total number of matches.

Limitations:
- Fuzzy matching may not always accurately capture the intended match. It relies on string similarity, which can be influenced by factors like typos and variations in spelling or formatting.
- The token sort ratio used for matching considers the order of tokens. If the order of words in the name and description doesn't match but the words are present, the match metric may not accurately reflect the intended match.

**Task 2: Finding Transactions with Similar Descriptions**

Solution:
- The solution utilizes language model embeddings to find transactions with semantically similar descriptions.
- It employs a pre-trained language model (BERT) and tokenizer to encode the input text and transaction descriptions into embeddings.
- The cosine similarity between the input embeddings and transaction embeddings is calculated to determine similarity.
- The transactions are sorted in descending order of similarity score to determine relevance.
- The API returns similar transactions, embeddings, and the total number of tokens used to create the embeddings.

Limitations:
- The accuracy of finding similar transactions heavily relies on the quality and relevance of the pre-trained language model used.
- Language model embeddings can be computationally expensive to calculate, especially with large datasets, resulting in slower response times for the API.
- Depending on the complexity of the language model and the number of transactions, memory usage may become a limitation.
- The solution assumes that a pre-trained language model is available and suitable for the task. Fine-tuning or using a domain-specific model might be necessary to achieve better results depending on the specific use case.

These limitations should be considered when applying the solution in real-world scenarios and further improvements could be explored based on specific requirements and constraints.
