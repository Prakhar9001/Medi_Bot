import os
import gdown

def download_model():
    """
    Downloads the Llama 2 model from Google Drive.
    """
    # This ID has been updated based on your Google Drive upload
    file_id = '144I4EHgc7HdnHQlkXyh_7oVJrLgQK8cG'
    url = f'https://drive.google.com/uc?id={file_id}'
    
    # Ensure the model directory exists
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created directory: {model_dir}")
        
    output_path = os.path.join(model_dir, 'llama-2-7b-chat.ggmlv3.q8_0-002.bin')
    
    if os.path.exists(output_path):
        print(f"Model already exists at: {output_path}")
        return

    print("Fetching the Llama 2 model (7GB) from Google Drive...")
    print("This may take 10-30 minutes depending on your internet speed.")
    
    try:
        gdown.download(url, output_path, quiet=False)
        print("\n✅ Download complete! You can now run the app.")
    except Exception as e:
        print(f"\n❌ Error downloading the model: {str(e)}")
        print("Please ensure the Google Drive link is public.")

if __name__ == "__main__":
    download_model()
