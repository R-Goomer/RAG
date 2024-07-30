# RAG Application with Streamlit

This project is a RAG (Retrieve and Generate) application built using Streamlit and LangChain. It allows users to upload multiple PDFs, select different embedding models (OpenAI, BAAI/BGE, Sentence Transformers), and interact with the content through a chat interface. The application utilizes Chroma VectorStore for managing and querying embeddings.


Try it HERE - https://huggingface.co/spaces/Rgoomer/rag_embedding

## Features

- **Upload PDFs**: Users can upload multiple PDF files.
- **Select Embedding Models**: Choose from BAAI/BGE embeddings, OpenAI embeddings, or Sentence Transformers.
- **Chroma VectorStore**: Efficiently stores and queries embeddings.
- **Chat Interface**: Engage with the uploaded PDFs through an interactive chat interface.

## Technologies Used

- **Streamlit**: For building the web application interface.
- **LangChain**: To handle language models and chaining operations.
- **OpenAI**: Provides pre-trained models for embeddings.
- **BAAI/BGE**: Provides embeddings through the BGE model.
- **Sentence Transformers**: Offers embeddings for sentences and documents.
- **Chroma VectorStore**: Manages and queries embeddings efficiently.

## Prerequisites

- Docker: Ensure Docker is installed on your machine. [Download Docker](https://www.docker.com/get-started).

## Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/your-repository.git
   cd your-repository
   ```

2. **Build the Docker Image**

   ```bash
   docker build -t rag-streamlit-app .
   ```

3. **Run the Docker Container**

   ```bash
   docker run -p 8501:8501 rag-streamlit-app
   ```

4. **Access the Application**

   Open your web browser and navigate to `http://localhost:8501` to start using the application.

## Project Structure

- `app.py`: Main Streamlit application script.
- `requirements.txt`: Lists Python dependencies required for the project.
- `Dockerfile`: Configuration file for building the Docker image.

## Usage

1. **Upload PDFs**: Use the upload feature to add your PDF files.
2. **Select Embedding Model**: Choose the desired embedding model from BAAI/BGE, OpenAI, or Sentence Transformers.
3. **Chroma VectorStore**: The app creates a Chroma VectorStore for storing and querying embeddings based on your selection.
4. **Chat with PDFs**: Use the chat interface to interact with and query the content of the uploaded PDFs.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your improvements or features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, reach out to [rushilgoomer@gmail.com](rushilgoomer@gmail.com).

Feel free to modify any sections to better fit your needs!