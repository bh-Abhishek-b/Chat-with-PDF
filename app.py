import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pickle
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback

with st.sidebar:
    st.title("📃Interactive PDF Chat App")
    st.markdown('''
    ## About
    This app is an LLM-powered chatbot built using:
    ''')
    st.markdown(' ## For Reference📑📚-')
    st.write('''-  [Streamlit](https://streamlit.io/)
            -  [LangChain](https://python.langchain.com/)
            -  [OpenAI](https://platform.openai.com/docs/models) LLM model
            ''')


def main():
     st.header("Chat with any PDF file 💬")
    
     load_dotenv()                                                                                               #  Setting up an envirnment for openai with authentication key
     pdf = st.file_uploader("Upload your PDF🔎", type='pdf')                                                     # To upload a pdf to the app


     if pdf is not None:
         pdf_file=PdfReader(pdf)
         file_name=pdf.name[:-4]
         st.write(file_name)
         content=''
         for i in pdf_file.pages:
             content+=i.extract_text()                                                                           # stores all the content of the pdf
        #  st.write(content)
         text_spiltter= RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len)    # defining the parameters for text splitting
         chuncks=text_spiltter.split_text(text=content)                                                          # Creating chuncks from the splitted text
         st.write(chuncks)

         if os.path.exists(f"{file_name}.pk1"):                                                                  # If the pdf file is already used, then use the same previous embeddings for cost effeciency
             with open(f'{file_name}.pk1','rb') as file:
                 Vector=pickle.load(file)
         else:
             embeddings=OpenAIEmbeddings()                                                                       # If a new file is used, then create new embeddings 
             Vector=FAISS.from_texts(chuncks,embedding=embeddings)
             with open(f'{file_name}.pk1','wb') as file:
                 pickle.dump(Vector,file)

         query = st.text_input("Ask question from your PDF file 🔎:")                                           # Questions regarding the PDF

         if query:
             docs=Vector.similarity_search(query=query,k=3)                                                     # Checking similarites in the vectorspace and the query
             llm=OpenAI() 
             chain=load_qa_chain(llm=llm,chain_type='stuff')
             with get_openai_callback() as callback:
                 response=chain.run(input_documents=docs,question=query)                                        # Generating Response from the llm about the query with best match from the simalarity search
                 print(callable)
             st.write(response) 



        
    
if __name__ == '__main__':
    main()