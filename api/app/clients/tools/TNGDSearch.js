// Import the necessary modules
const { Pinecone } = require('@pinecone-database/pinecone');

const { Tool } = require('langchain/tools');
const { z } = require('zod');
const { logger } = require('~/config');
const { OpenAIClient, AzureKeyCredential } = require('@azure/openai');

class TNGDSearch extends Tool {
  // Constants for default values and configurations
  static DEFAULT_PINECONE_INDEX = 'large-index';

  constructor(fields = {}) {
    super();
    this.name = 'pinecone';
    this.description =
      'Retrieve relevant information from a Pinecone vector database based on query embeddings.';

    // Initialize Pinecone client configurations using fields or environment variables
    this.pineconeApiKey = fields.PINECONE_PROJECT_API_KEY || process.env.PINECONE_PROJECT_API_KEY;
    this.pineconeIndex =
      fields.PINECONE_INDEX_NAME ||
      process.env.PINECONE_INDEX_NAME ||
      TNGDSearch.DEFAULT_PINECONE_INDEX;
    this.openAIApiKey = fields.AZURE_API_KEY || process.env.AZURE_API_KEY;
    this.openAIHost = fields.AZURE_HOST || process.env.AZURE_HOST;

    this.openAIClient = new OpenAIClient(
      this.openAIHost,
      new AzureKeyCredential(this.openAIApiKey),
    );

    if (!this.pineconeApiKey) {
      throw new Error('Missing PINECONE_PROJECT_API_KEY environment variable.');
    }

    if (!this.openAIApiKey) {
      throw new Error('Missing OPENAI_API_KEY environment variable.');
    }

    // Instantiate PineconeClient
    this.client = new Pinecone({
      apiKey: this.pineconeApiKey,
    });
    logger.info(`Pinecone client: ${this.client}`); // Logging the Pinecone client instance
    logger.info(`Pinecone index name: ${this.pineconeIndex}`);

    this.index = this.client.index(this.pineconeIndex);
    logger.info(`Pinecone index: ${this.index}`); // Logging the Pinecone index instance

    // Define schema for input validation
    this.schema = z.object({
      query: z.string().describe('Query text to convert into embeddings and search in Pinecone'),
    });
  }

  async queryToEmbeddings(query) {
    // Generate embeddings for the query using OpenAI client
    const data = await this.openAIClient.getEmbeddings('text-embedding-3-large', query);

    return data.data[0].embedding;
  }

  async _call(data) {
    const { query } = this.schema.parse(data); // Validate and extract the query from data using Zod schema

    try {
      const embeddings = await this.queryToEmbeddings(query);
      // Use the Pinecone client to search using embeddings. Assuming `search` is a method provided by the Pinecone client library.
      const results = await this.index.query({
        vector: embeddings,
        topK: 3,
        includeMetadata: true,
      });

      let resultString = '';
      // loop through results.matches
      let firstResult = null;
      for (const result of results.matches) {
        if (!firstResult) {
          firstResult = result;
        }
        resultString += `Page Title: ${result.metadata.page_title} Content: ${result.metadata.content} `;
      }
      if (!firstResult) {
        return 'No results found for the query.';
      }
      return resultString;
    } catch (error) {
      logger.error('Pinecone search request failed', error); // Logging the error using a configured logger
      return 'There was an error with the Pinecone search.';
    }
  }
}

// Export the plugin
module.exports = TNGDSearch;
