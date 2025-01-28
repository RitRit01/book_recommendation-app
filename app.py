import streamlit as st
import torch
import pickle
import pandas as pd
from torch import nn
import os
import requests

# Collaborative Filtering Model
class CollaborativeFilteringModel(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim, hidden_dim):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)

        self.hidden_layer1 = nn.Linear(embedding_dim * 2, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)  # Add batch normalization
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)  # Add batch normalization

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim // 2, 1)

    def forward(self, user_indices, item_indices):
        user_embedded = self.user_embedding(user_indices)
        item_embedded = self.item_embedding(item_indices)
        user_bias = self.user_bias(user_indices).squeeze()
        item_bias = self.item_bias(item_indices).squeeze()

        concatenated = torch.cat([user_embedded, item_embedded], dim=1)
        hidden_output1 = self.relu(self.batch_norm1(self.hidden_layer1(concatenated)))
        hidden_output2 = self.relu(self.batch_norm2(self.hidden_layer2(hidden_output1)))
        output = self.output_layer(hidden_output2).squeeze() + user_bias + item_bias
        return output

# Content-Based Filtering Model
class ContentBasedFilteringModel(nn.Module):
    def __init__(self, num_categories, num_authors, num_titles, embedding_dim):
        super(ContentBasedFilteringModel, self).__init__()
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        self.author_embedding = nn.Embedding(num_authors, embedding_dim)
        self.title_embedding = nn.Embedding(num_titles, embedding_dim)

        self.hidden_layer1 = nn.Linear(4 * embedding_dim, 128)
        self.hidden_layer2 = nn.Linear(128, 64)
        self.output_layer = nn.Linear(64, 1)

        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()

    def forward(self, category_indices, author_indices, title_indices, sentiment_scores):
        category_embedded = self.category_embedding(category_indices)
        author_embedded = self.author_embedding(author_indices)
        title_embedded = self.title_embedding(title_indices)

        sentiment_expanded = sentiment_scores.unsqueeze(1).expand_as(category_embedded)
        concatenated = torch.cat([category_embedded, author_embedded, title_embedded, sentiment_expanded], dim=1)

        hidden_output1 = self.relu(self.hidden_layer1(concatenated))
        hidden_output2 = self.relu(self.hidden_layer2(hidden_output1))
        output = self.output_layer(hidden_output2).squeeze()
        return output

# Helper function to modify Google Drive links
def get_direct_download_url(drive_url):
    """
    Convert Google Drive sharing link to a direct download link.
    """
    if "drive.google.com" in drive_url:
        file_id = drive_url.split("/d/")[1].split("/")[0]
        return f"https://drive.google.com/uc?export=download&id={file_id}"
    else:
        return drive_url

def download_file_from_drive(drive_url, output_path):
    if not os.path.exists(output_path):
        direct_url = get_direct_download_url(drive_url)
        st.info(f"Downloading {output_path} from Google Drive...")
        response = requests.get(direct_url, stream=True)
        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        st.success(f"{output_path} downloaded successfully!")
    return output_path

@st.cache_resource
def load_models_and_data():
    # Define Google Drive links
    dataset_url = "https://drive.google.com/file/d/1NfqAOGslnIT8wQYa8ZWLgzqeuoJwkDDs/view?usp=sharing"
    cf_model_url = "https://drive.google.com/file/d/1to_cmpPNMnZspZtskdOuuYbVZNsF70Gz/view?usp=sharing"
    cbf_model_url = "https://drive.google.com/file/d/1brAu54LbjTEeUBmu1z_kpD4E2EAINtHV/view?usp=sharing"

    # File paths
    dataset_path = "data_with_sentiment_labels.csv"
    cf_model_path = "cf_model.pth"
    cbf_model_path = "cbf_model.pth"

    # Download files if they don't exist locally
    download_file_from_drive(dataset_url, dataset_path)
    download_file_from_drive(cf_model_url, cf_model_path)
    download_file_from_drive(cbf_model_url, cbf_model_path)

    # Load dataset
    data = pd.read_csv(dataset_path, on_bad_lines="skip")  # Skip malformed rows

    # Strip spaces from column names
    data.columns = data.columns.str.strip()

    # Validate required columns
    required_columns = {"User_id", "Id", "Title"}
    if not required_columns.issubset(data.columns):
        st.error(f"Dataset is missing required columns: {required_columns - set(data.columns)}")
        return None, None, None, None, None, None

    # Re-encode the dataset
    data, user_encoder, book_encoder, title_encoder = encode_labels_full_dataset(data)

    # Save the updated encoders
    with open("user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)
    with open("book_encoder.pkl", "wb") as f:
        pickle.dump(book_encoder, f)
    with open("title_encoder.pkl", "wb") as f:
        pickle.dump(title_encoder, f)

    # Load CF Model
    cf_model = CollaborativeFilteringModel(len(user_encoder), len(book_encoder), embedding_dim=150, hidden_dim=128)
    cf_model.load_state_dict(torch.load(cf_model_path, map_location=torch.device("cpu")))
    cf_model.eval()

    # Load CBF Model
    checkpoint = torch.load(cbf_model_path, map_location=torch.device("cpu"))
    cbf_model = ContentBasedFilteringModel(
        num_categories=checkpoint['category_embedding.weight'].shape[0],
        num_authors=checkpoint['author_embedding.weight'].shape[0],
        num_titles=len(title_encoder),
        embedding_dim=150,
    )
    cbf_model.load_state_dict(checkpoint)
    cbf_model.eval()

    return data, cf_model, cbf_model, user_encoder, book_encoder, title_encoder

@st.cache_resource
def create_reverse_mappings(book_encoder, title_encoder):
    reverse_book_encoder = {v: k for k, v in book_encoder.items()}
    reverse_title_index = {v: k for k, v in title_encoder.items()}
    return reverse_book_encoder, reverse_title_index

def recommend_for_user(user_id, cf_model, cbf_model, user_encoder, book_encoder, reverse_book_encoder, reverse_title_index, is_new_user=False, top_n=15):
    all_books = list(book_encoder.values())
    max_title_index = cbf_model.title_embedding.num_embeddings

    valid_books = [idx for idx in all_books if idx < max_title_index]

    with torch.no_grad():
        user_tensor = torch.tensor([user_encoder.get(user_id, 0)] * len(valid_books), dtype=torch.long)
        book_tensor = torch.tensor(valid_books, dtype=torch.long)
        cf_predictions = cf_model(user_tensor, book_tensor).squeeze().numpy()

        category_tensor = torch.tensor([0] * len(valid_books), dtype=torch.long)
        author_tensor = torch.tensor([0] * len(valid_books), dtype=torch.long)
        title_tensor = torch.tensor(valid_books, dtype=torch.long)
        sentiment_tensor = torch.tensor([0.5] * len(valid_books), dtype=torch.float32)
        cbf_predictions = cbf_model(category_tensor, author_tensor, title_tensor, sentiment_tensor).squeeze().numpy()

        hybrid_scores = 0.5 * cf_predictions + 0.5 * cbf_predictions
        top_indices = hybrid_scores.argsort()[-top_n:][::-1]

        recommendations = [
            {
                "Book ID": reverse_book_encoder[valid_books[idx]],
                "Title": reverse_title_index[valid_books[idx]],
                "Hybrid Score": hybrid_scores[idx],
            }
            for idx in top_indices if valid_books[idx] in reverse_title_index
        ]

    return recommendations

def main():
    st.title("Optimized Hybrid Book Recommendation System")

    data, cf_model, cbf_model, user_encoder, book_encoder, title_encoder = load_models_and_data()
    reverse_book_encoder, reverse_title_index = create_reverse_mappings(book_encoder, title_encoder)

    user_id = st.text_input("Enter User ID:")

    if st.button("Get Recommendations"):
        is_new_user = user_id not in user_encoder
        recommendations = recommend_for_user(
            user_id, cf_model, cbf_model, user_encoder, book_encoder, reverse_book_encoder, reverse_title_index, is_new_user
        )

        st.subheader(f"Top Recommendations for {'New' if is_new_user else 'Existing'} User: {user_id}")
        for rec in recommendations:
            st.write(f"Book ID: {rec['Book ID']}, Title: {rec['Title']}, Hybrid Score: {rec['Hybrid Score']:.2f}")

if __name__ == "__main__":
    main()
