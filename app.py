import streamlit as st
import torch
import pickle
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset

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


# Re-encode the dataset and save encoders
def encode_labels_full_dataset(data):
    data['user_encoded'], user_classes = pd.factorize(data['User_id'])
    data['book_encoded'], book_classes = pd.factorize(data['Id'])
    data['title_encoded'], title_classes = pd.factorize(data['Title'])

    user_encoder = {label: idx for idx, label in enumerate(user_classes)}
    book_encoder = {label: idx for idx, label in enumerate(book_classes)}
    title_encoder = {label: idx for idx, label in enumerate(title_classes)}

    return data, user_encoder, book_encoder, title_encoder

def ensure_embedding_dimensions(cbf_model, title_to_index, checkpoint):
    """
    Adjust the embedding dimensions of the ContentBasedFilteringModel to match the current dataset dimensions.
    """
    num_titles = len(title_to_index)
    title_size = checkpoint['title_embedding.weight'].shape[0]

    if title_size != num_titles:
        st.warning(f"Adjusting title embedding size from {title_size} to {num_titles}.")
        cbf_model.title_embedding = nn.Embedding(num_titles, cbf_model.title_embedding.embedding_dim)

    # Update the model's state_dict with compatible weights
    model_dict = cbf_model.state_dict()
    checkpoint_dict = {k: v for k, v in checkpoint.items() if k in model_dict and v.shape == model_dict[k].shape}
    model_dict.update(checkpoint_dict)
    cbf_model.load_state_dict(model_dict)

    return cbf_model

def download_file_from_drive(drive_url, output_path):
    if not os.path.exists(output_path):
        st.info(f"Downloading {output_path} from Google Drive...")
        response = requests.get(drive_url)
        with open(output_path, "wb") as f:
            f.write(response.content)
        st.success(f"{output_path} downloaded successfully!")
    return output_path

# Update `load_models_and_data` to download models
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
    data = pd.read_csv(dataset_path)

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
    num_users = len(user_encoder)
    num_books = len(book_encoder)
    cf_model = CollaborativeFilteringModel(num_users, num_books, embedding_dim=150, hidden_dim=128)
    cf_model.load_state_dict(torch.load(cf_model_path, map_location=torch.device("cpu")))
    cf_model.eval()

    # Load CBF Model
    checkpoint = torch.load(cbf_model_path, map_location=torch.device("cpu"))
    cbf_model = ContentBasedFilteringModel(
        num_categories=checkpoint['category_embedding.weight'].shape[0],
        num_authors=checkpoint['author_embedding.weight'].shape[0],
        num_titles=len(title_encoder),  # Updated to match current dataset
        embedding_dim=150,
    )
    cbf_model = ensure_embedding_dimensions(cbf_model, title_encoder, checkpoint)
    cbf_model.eval()

    return data, cf_model, cbf_model, user_encoder, book_encoder, title_encoder


# Reverse mappings
@st.cache_resource
def create_reverse_mappings(book_encoder, title_encoder):
    reverse_book_encoder = {v: k for k, v in book_encoder.items()}
    reverse_title_index = {v: k for k, v in title_encoder.items()}
    return reverse_book_encoder, reverse_title_index


# Recommendations
def recommend_for_user(user_id, cf_model, cbf_model, user_encoder, book_encoder, reverse_book_encoder, reverse_title_index, is_new_user=False, top_n=15):
    all_books = list(book_encoder.values())
    max_title_index = cbf_model.title_embedding.num_embeddings

    # Filter valid book indices
    valid_books = [idx for idx in all_books if idx < max_title_index]

    with torch.no_grad():
        # CF Predictions
        user_tensor = torch.tensor([user_encoder.get(user_id, 0)] * len(valid_books), dtype=torch.long)
        book_tensor = torch.tensor(valid_books, dtype=torch.long)
        cf_predictions = cf_model(user_tensor, book_tensor).squeeze().numpy()

        # CBF Predictions
        category_tensor = torch.tensor([0] * len(valid_books), dtype=torch.long)
        author_tensor = torch.tensor([0] * len(valid_books), dtype=torch.long)
        title_tensor = torch.tensor(valid_books, dtype=torch.long)
        sentiment_tensor = torch.tensor([0.5] * len(valid_books), dtype=torch.float32)
        cbf_predictions = cbf_model(category_tensor, author_tensor, title_tensor, sentiment_tensor).squeeze().numpy()

        # Hybrid Scores
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


# Streamlit App
def main():
    st.title("Optimized Hybrid Book Recommendation System")

    # Load models, data, and encoders
    data, cf_model, cbf_model, user_encoder, book_encoder, title_encoder = load_models_and_data()
    reverse_book_encoder, reverse_title_index = create_reverse_mappings(book_encoder, title_encoder)

    # User input
    user_id = st.text_input("Enter User ID:")

    if st.button("Get Recommendations"):
        is_new_user = user_id not in user_encoder
        recommendations = recommend_for_user(
            user_id, cf_model, cbf_model, user_encoder, book_encoder, reverse_book_encoder, reverse_title_index, is_new_user
        )

        # Display recommendations
        st.subheader(f"Top Recommendations for {'New' if is_new_user else 'Existing'} User: {user_id}")
        for rec in recommendations:
            st.write(f"Book ID: {rec['Book ID']}, Title: {rec['Title']}, Hybrid Score: {rec['Hybrid Score']:.2f}")


if __name__ == "__main__":
    main()
