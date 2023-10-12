package com.datastax.rag;

import static com.dtsx.astra.sdk.utils.TestUtils.getAstraToken;
import static com.dtsx.astra.sdk.utils.TestUtils.setupDatabase;
import static dev.langchain4j.model.openai.OpenAiModelName.GPT_3_5_TURBO;
import static dev.langchain4j.model.openai.OpenAiModelName.TEXT_EMBEDDING_ADA_002;
import static java.time.Duration.ofSeconds;
import static java.util.stream.Collectors.joining;

import java.io.File;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.boot.autoconfigure.cassandra.CassandraAutoConfiguration;
import org.springframework.boot.autoconfigure.data.cassandra.CassandraDataAutoConfiguration;

import com.dtsx.astra.sdk.utils.TestUtils;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.DocumentType;
import dev.langchain4j.data.document.FileSystemDocumentLoader;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.model.chat.ChatLanguageModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.input.Prompt;
import dev.langchain4j.model.input.PromptTemplate;
import dev.langchain4j.model.openai.OpenAiChatModel;
import dev.langchain4j.model.openai.OpenAiEmbeddingModel;
import dev.langchain4j.model.openai.OpenAiTokenizer;
import dev.langchain4j.model.output.Response;
import dev.langchain4j.store.embedding.EmbeddingMatch;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.EmbeddingStoreIngestor;
import dev.langchain4j.store.embedding.cassandra.AstraDbEmbeddingStore;

@SpringBootApplication(exclude = {
        CassandraDataAutoConfiguration.class,
        CassandraAutoConfiguration.class }
)
public class SpringChatbotApplication implements CommandLineRunner {

	private static final java.util.logging.Logger log = java.util.logging.Logger.getLogger(SpringChatbotApplication.class.getName());
	
    @Value("${astra.database.name}")
    private static String databaseName = "langchain4j";

    @Value("${astra.database.keyspace}")
    private static String keyspace = "langchain4j";

    @Value("${astra.database.table}")
    private static String tableName = "table_story";

    private static String astraToken;
    private static String openAIKey;
    @Value("${astra.api.database-id}")
    private static String databaseId;
    private static EmbeddingModel embeddingModel;
    private static EmbeddingStore<TextSegment> embeddingStore;

	public static void main(String[] args) {
		SpringApplication.run(SpringChatbotApplication.class, args);
	}

    public void run(String... args) 
    {
        astraToken = getAstraToken();
        openAIKey = System.getenv("OPENAI_API_KEY");
        databaseId = System.getenv("ASTRA_DB_ID");
        		
        log.info("Pre-requisites are met.");

        // Embedding model (OpenAI)
        embeddingModel = OpenAiEmbeddingModel.builder()
                .apiKey(openAIKey)
                .modelName(TEXT_EMBEDDING_ADA_002)
                .timeout(ofSeconds(15))
                .logRequests(true)
                .logResponses(true)
                .build();
        log.info("[OK] - Embeddings Model (OpenAI)");

        // Embed the document and it in the store
        embeddingStore = AstraDbEmbeddingStore.builder()
                .token(astraToken)
                .database(databaseId, TestUtils.TEST_REGION)
                .table(keyspace, tableName)
                .vectorDimension(1536)
                .build();
        log.info("[OK] - Embeddings Store (Astra)");
 
        Path path = new File(getClass().getResource("/usa.txt").getFile()).toPath();
        Document document = FileSystemDocumentLoader.loadDocument(path, DocumentType.TXT);
        DocumentSplitter splitter = DocumentSplitters
                .recursive(100, 10,
                        new OpenAiTokenizer(GPT_3_5_TURBO));
        log.info("[OK] - Document SPLITTER");

        // Ingest method 2
        EmbeddingStoreIngestor.builder()
                .documentSplitter(splitter)
                .embeddingModel(embeddingModel)
                .embeddingStore(embeddingStore)
                .build().ingest(document);
        log.info("[OK] - DOCUMENT PROCESSED");

        // Specify the question you want to ask the model
        String question = "Which documents do Germans require to enter the USA?";

        // Embed the question
        Response<Embedding> questionEmbedding = embeddingModel.embed(question);

        // Find relevant embeddings in embedding store by semantic similarity
        // You can play with parameters below to find a sweet spot for your specific use case
        int maxResults = 3;
        double minScore = 0.8;
        List<EmbeddingMatch<TextSegment>> relevantEmbeddings =
                embeddingStore.findRelevant(questionEmbedding.content(), maxResults, minScore);

        // --------- Chat Template -------------

        // Create a prompt for the model that includes question and relevant embeddings
        PromptTemplate promptTemplate = PromptTemplate.from(
                "Answer the following question to the best of your ability:\n"
                        + "\n"
                        + "Question:\n"
                        + "{{question}}\n"
                        + "\n"
                        + "Base your answer on the following information:\n"
                        + "{{information}}");

        String information = relevantEmbeddings.stream()
                .map(match -> match.embedded().text())
                .collect(joining("\n\n"));

        Map<String, Object> variables = new HashMap<>();
        variables.put("question", question);
        variables.put("information", information);

        Prompt prompt = promptTemplate.apply(variables);

        // Send the prompt to the OpenAI chat model
        ChatLanguageModel chatModel = OpenAiChatModel.builder()
                .apiKey(openAIKey)
                .modelName(GPT_3_5_TURBO)
                .temperature(0.7)
                .timeout(ofSeconds(15))
                .maxRetries(3)
                .logResponses(true)
                .logRequests(true)
                .build();

        Response<AiMessage> aiMessage = chatModel.generate(prompt.toUserMessage());

        // See an answer from the model
        String answer = aiMessage.content().text();
        System.out.println(answer);
    }
}
