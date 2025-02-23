# Ignore warnings
import sqlite3

from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes
from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models.utils.enums import DecodingMethods
import warnings

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.embeddings import HuggingFaceEmbeddings, SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_ibm import WatsonxEmbeddings

from menTHOR import settings

warnings.filterwarnings("ignore")
wxa_url = "https://us-south.ml.cloud.ibm.com"
wxa_api_key = settings.wxa_api_key
wxa_project_id = settings.wxa_project_id




from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
#granite_llm_ibm = WatsonxLLM(model=model)

#response = granite_llm_ibm(query)

#print(response)
#filename = "/Users/michelemenabeni/PycharmProjects/menTHOR/menTHOR/IBM-Granite-AI-Hackathon-2025.pdf"
from langchain.document_loaders import PDFPlumberLoader

from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import SQLiteVSS

#db_path = settings.DATABASES['default']['NAME']
#watsonx_llm = WatsonxLLM(model=model)
# Connessione al database
#connection = sqlite3.connect(db_path)
# Crea il vettore store utilizzando la connessione al database Django
'''vectorstore = SQLiteVSS.from_documents(
    documents=texts,
    embedding=emb_func,
    connection=connection,  # Passa la connessione al database
    table_name="document_vectors"  # Nome della tabella per i vettori
)'''
#loader = PDFPlumberLoader('/Users/michelemenabeni/PycharmProjects/menTHOR/templates/documents/ibm_annual_report_2023.pdf')
#documents = loader.load()

#text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
#texts = text_splitter.split_documents(documents)
embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}
emb_func = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=wxa_project_id,
    params=embed_params,
    apikey = wxa_api_key,
)
#emb_func = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-107m-multilingual")
# Percorso al database SQLite di Django
#vectorstore = FAISS.from_documents(documents=texts, embedding=emb_func)

#vectorstore.save_local("faiss_index")
'''embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {"input_text": True},
}
emb_func = WatsonxEmbeddings(
    model_id="ibm/slate-125m-english-rtrvr",
    url="https://us-south.ml.cloud.ibm.com",
    project_id=wxa_project_id,
    params=embed_params,
    apikey = wxa_api_key,
)'''
parameters = {
        GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
        GenParams.MIN_NEW_TOKENS: 1,
        GenParams.MAX_NEW_TOKENS: 300
    }
model = Model(
        model_id="ibm/granite-20b-multilingual",
        params=parameters,
        credentials={
            "url": wxa_url,
            "apikey": wxa_api_key
        },
        project_id=wxa_project_id)
watsonx_llm = WatsonxLLM(model=model)
vectorstore = FAISS.load_local("/Users/michelemenabeni/PycharmProjects/menTHOR/menTHOR/faiss_index", embeddings=emb_func,
        allow_dangerous_deserialization=True)
# Carica il vectorstore da disco
def get_answer(query):


    #watsonx_llm = WatsonxLLM(model=model)
    #emb_func = HuggingFaceEmbeddings(model_name="ibm-granite/granite-embedding-107m-multilingual")

    #results = vectorstore.similarity_search(query, k=3)
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Answer to the question using this context, without providing additional questions:\n\n{context}\n\nQuestion: {question}"
    )
    qa = RetrievalQA.from_chain_type(
        llm=watsonx_llm,  # Modello di linguaggio
        chain_type="stuff",  # Tipo di catena
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),  # Retriever con 5 risultati
        return_source_documents=True,  # Restituisci i documenti sorgente
        chain_type_kwargs={"prompt": prompt_template}  # Usa il prompt template personalizzato
    )

    result = qa.invoke({"query": query})
    print(result)
    answer = result["result"]  # La risposta generata dal modello
    source_documents = result["source_documents"]  # I documenti sorgente utilizzati
    return answer, source_documents[0].page_content
#print(get_answer("What is watsonX?"))



