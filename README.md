Hi, my project title is Fake News Detection using DistilBERT.

In this project, I used DistilBERT, which is a pre-trained transformer model from Hugging Face, for binary text classification to identify whether a given news article is real or fake.

I collected my dataset from Kaggle, which contains cinema and entertainment-related news articles, labeled as real and fake.

Since DistilBERT is a transformer-based model, it does not require extensive text preprocessing like traditional machine learning models. Therefore, I performed only minimal preprocessing, such as removing URLs, special characters, and handling missing values, without applying stopword removal or stemming.

After preprocessing, I split the dataset into training and testing sets using an 80–20 split for model evaluation.

Next, I initialized the DistilBERT tokenizer and the DistilBERT sequence classification model using Hugging Face’s Transformers library. I then tokenized the text data into input IDs and attention masks.

To handle the tokenized data efficiently, I created a custom PyTorch Dataset class, which converts the tokenized inputs and labels into tensors suitable for training.

Using Hugging Face’s Trainer API, I fine-tuned the pre-trained DistilBERT model on my dataset. After training, I evaluated the model using accuracy on the test dataset and saved the fine-tuned model locally.

For deployment, I developed a FastAPI backend, which loads the trained model and exposes an API endpoint to accept news text and return predictions in real time.

I also created a frontend using HTML and CSS, which allows users to input news text through a web interface. This frontend communicates with the FastAPI backend to display whether the news is real or fake along with a confidence score.

Finally, I deployed the entire application on Hugging Face Spaces using Docker. I created a Dockerfile to run the FastAPI application on port 7860, included all dependencies in a requirements file, and uploaded the model, backend, and frontend files to Hugging Face. Once deployed, the application became accessible through a public URL.
